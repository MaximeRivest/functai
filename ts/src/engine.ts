/**
 * One call of an AI function: lay it out (lmcc), send it (lm15), read the
 * reply (lmcc), run tools until the model answers (contract/functions.md,
 * "When the reply cannot be read"). No prompt text is written here except
 * the one re-ask sentence the contract gives; every other byte of the request
 * comes from the plan.
 */

import { cacheOf, forget, keep, lookup, replyKey } from "./cache.ts";
import * as lmcc from "lmcc";
import * as bridge from "lmcc/lm15";
import { Delta, Message, RETRYABLE_ERRORS, materializeResponse, responseToEvents, type Config, type Request, type Response,
  type StreamEvent } from "@lm15/lm15";
import type { Call } from "./calllog.ts";
import { misfit } from "./shapes.ts";
import { entriesOf, setOwn, writeData } from "./values.ts";
import { configOf, type Settings } from "./settings.ts";
import { prepareInputs } from "./signature.ts";

type Rec = Record<string, unknown>;

/** A tool the model may call: its name, what it does, the JSON Schema of its input, and the code. */
export interface Tool {
  readonly name: string;
  readonly description?: string;
  readonly parameters: Rec;
  readonly run: (input: any) => unknown;
}

/** The tool loop reached `maxSteps` without an answer. */
export class StepLimit extends Error {
  readonly turn: lmcc.Turn;
  constructor(message: string, turn: lmcc.Turn) {
    super(message);
    this.turn = turn;
    this.name = "StepLimit";
  }
}

/** A stream closed the call. */
export class Cancelled extends Error {
  constructor(message = "the stream was closed") {
    super(message);
    this.name = "Cancelled";
  }
}

/** Everything one call produced. `answer` is the value a call returns. */
export class Prediction<O = Rec, A = unknown> {
  /** Every output, by name. */
  readonly outputs: O;
  readonly answerName: string;
  /** The call log's id of this call: rate it with `rate(prediction, "right")`. */
  readonly callId: string;
  /** The lmcc turn: inputs, every model step and tool result, outputs. */
  readonly turn: lmcc.Turn;
  readonly response: Response | null;
  readonly responses: readonly Response[];
  readonly repairs: readonly unknown[];

  constructor(outputs: O, answerName: string, callId: string, turn: lmcc.Turn, response: Response | null,
    responses: readonly Response[], repairs: readonly unknown[] = []) {
    this.outputs = outputs;
    this.answerName = answerName;
    this.callId = callId;
    this.turn = turn;
    this.response = response;
    this.responses = responses;
    this.repairs = repairs;
  }

  get answer(): A {
    return (this.outputs as Rec)[this.answerName] as A;
  }

  get attempts(): number {
    return this.responses.length;
  }
}

export interface Router {
  complete(request: Request, opts?: { signal?: AbortSignal }): Promise<Response>;
  stream?(request: Request, opts?: { signal?: AbortSignal }): AsyncIterable<StreamEvent>;
}

export interface Job {
  readonly function: string;
  readonly plan: lmcc.Plan;
  readonly past: readonly lmcc.Turn[];
  readonly inputs: Rec;
  readonly settings: Settings & { retries: number; apiRetries: number; maxSteps: number; toolErrors: string };
  readonly router: Router;
  readonly model: string;
  readonly tools: readonly Tool[];
  readonly call: Call;
  readonly answer: string;
}

const sleep = (ms: number, signal?: AbortSignal) => new Promise<void>((resolve, reject) => {
  if (signal?.aborted) {
    reject(new Cancelled());
    return;
  }
  const onAbort = () => {
    clearTimeout(t);
    reject(new Cancelled());
  };
  const t = setTimeout(() => {
    signal?.removeEventListener("abort", onAbort);
    resolve();
  }, ms);
  signal?.addEventListener("abort", onAbort, { once: true });
});

const retryable = (err: unknown) => RETRYABLE_ERRORS.some((cls) => err instanceof cls);

/** A request event: the call begins a request to a model (streaming.md, law 8: each is an exchange). */
function requested(job: Job): void {
  job.call.requests += 1;
  job.call.event("request", { request: job.call.requests, model: job.model });
}

const stopped = (job: Job) => {
  if (job.call.signal?.aborted) throw new Cancelled();
};

/** One request: from the reply cache when on, else through the router, re-sent after transient errors; streamed when watched. */
async function send(job: Job, request: Request, requestHash: string): Promise<Response> {
  const { call } = job;
  const signal = call.signal;
  const cache = cacheOf(job.settings.cacheReplies);
  const key = cache ? replyKey(request) : null;
  if (cache && key) {
    stopped(job);
    const hit = await lookup(cache, key);
    if (hit) {
      requested(job);
      call.exchange(job.model, request, hit, Date.now(), 0, { cached: true, requestHash });
      if (call.watched) replay(job, hit);           // a whole reply: one text piece per field (streaming.md)
      return hit;
    }
  }
  const retries = Math.max(0, job.settings.apiRetries);
  for (let attempt = 0; ; attempt++) {
    stopped(job);
    const started = Date.now();
    const t0 = performance.now();
    requested(job);
    const watched = call.watched;
    try {
      let response: Response;
      let first: number | null = null;
      if (watched && job.router.stream) {
        // The view (lmcc's stream reader) never decides the call: a piece it
        // cannot read stops the view, and the whole reply is read below.
        const events: StreamEvent[] = [];
        const view = job.plan.stream();
        let viewing = true;
        const thinkingIsRead = job.plan.signature.fields.some((f) => f.purpose === "reasoning");
        const show = (batch: readonly { kind: string; field?: string; text?: string }[]) => {
          for (const ev of batch) if (ev.kind === "field_delta" && ev.field !== "calls" && ev.text) text(job, ev.field!, ev.text);
        };
        for await (const e of job.router.stream(request, signal ? { signal } : undefined)) {
          stopped(job);
          events.push(e);
          if (e.type !== "delta") continue;
          first ??= (performance.now() - t0) / 1000;
          const d = e.delta as { type: string; text?: string };
          if (d.type === "thinking" && !thinkingIsRead && d.text) call.event("thinking", { text: d.text });
          if (viewing) {
            try {
              show(view.feed(Delta.toJSON(e.delta) as Rec) as never);
            } catch {
              viewing = false;
            }
          }
        }
        if (viewing) {
          const end = events.find((e) => e.type === "end") as { finishReason?: string } | undefined;
          try {
            show(view.finish(end?.finishReason ?? null).events as never);
          } catch {
            viewing = false;
          }
        }
        response = materializeResponse(events, request);
      } else {
        response = await job.router.complete(request, signal ? { signal } : undefined);
        if (watched) replay(job, response);
      }
      call.exchange(job.model, request, response, started, (performance.now() - t0) / 1000,
        { streamed: watched && job.router.stream !== undefined, firstDelta: first, requestHash });
      if (cache && key && response.finishReason !== "error") await keep(cache, key, response);
      return response;
    } catch (err) {
      call.exchange(job.model, request, null, started, (performance.now() - t0) / 1000,
        { error: err, streamed: watched && job.router.stream !== undefined, requestHash });
      if (signal?.aborted) throw new Cancelled();
      if (!retryable(err) || attempt >= retries) throw err;
      const after = (err as { retryAfter?: number }).retryAfter;
      const wait = typeof after === "number" && after > 0 ? after : Math.min(30, 2 ** attempt) * (0.5 + Math.random());
      call.event("retry", { reason: `the provider failed (${(err as Error).name}); sending again in ${wait.toFixed(1)} s`, wait });
      await sleep(wait * 1000, signal);
    }
  }
}

/** A piece of an output's text (the answer's when it is the answer field). */
function text(job: Job, field: string, piece: string): void {
  job.call.event("text", { field, answer: field === job.answer, text: piece });
}

/** A reply that came whole, shown as one text piece per field (streaming.md, law 6). */
function replay(job: Job, response: Response): void {
  const s = job.plan.stream();
  const shown: Array<{ kind: string; field?: string; text?: string }> = [];
  try {
    for (const e of responseToEvents(response)) {
      if (e.type === "delta") shown.push(...(s.feed(e.delta as unknown as Rec) as typeof shown));
    }
    shown.push(...(s.finish(response.finishReason ?? null).events as typeof shown));
  } catch {
    return;                                   // an unreadable reply: the re-ask will say so
  }
  const texts = new Map<string, string>();
  for (const e of shown) if (e.kind === "field_delta" && e.field !== "calls") texts.set(e.field!, (texts.get(e.field!) ?? "") + e.text);
  for (const [field, piece] of texts) if (piece) text(job, field, piece);
}

/**
 * The lm15 request for a rendered plan: lmcc's bridge, the plan's request
 * settings with the caller's Config merged under them. The bridge hands lm15
 * both as data it takes (lmcc D-59): member order kept where lm15 keeps it
 * (lmcc's record is lm15's), a big integer as lm15's own number.
 */
export function lm15Request(rendered: lmcc.RenderResult, model: string, config: Config | undefined): Request {
  return bridge.request(rendered, { model, config });
}

/** The first output value that does not fit its shape, as a parse-value refusal. */
function checkValues(plan: lmcc.Plan, values: Rec): void {
  for (const f of plan.signature.fields) {
    if (f.direction !== "output" || f.purpose !== "plain" || !Object.hasOwn(values, f.name)) continue;
    const problem = misfit(f.shape as Rec, values[f.name], f.name);
    if (problem) throw new lmcc.Refusal("parse-value", problem);
  }
}

function askedAgain(err: lmcc.Refusal): string {
  return err.code === "parse-truncated"
    ? "the reply was cut off; asking again with a larger token budget"
    : `the reply could not be read (${err.hint}); asking again`;
}

/** One model call; after an unreadable reply, up to `retries` follow-ups that send the reader's hint back. */
async function complete(job: Job, rendered: lmcc.RenderResult, responses: Response[]): Promise<[Response, lmcc.Reading]> {
  const overrides: Rec = {};
  let request = lm15Request(rendered, job.model, configOf(job.settings));
  // lmcc's hash of the request each exchange sends (calls.md, exchanges): the render's; a re-ask's is the one it follows
  // with the reply's message and the correction appended; a re-send with a larger token budget sends the same lmcc request
  let asked = rendered.request() as Rec;
  let requestHash = lmcc.sha256(asked as lmcc.Json);
  const retries = Math.max(0, job.settings.retries);
  for (let attempt = 0; ; attempt++) {
    const response = await send(job, request, requestHash);
    responses.push(response);
    try {
      const reading = bridge.read(job.plan, response);
      checkValues(job.plan, reading.values as Rec);
      return [response, reading];
    } catch (err) {
      if (!lmcc.isRefusal(err)) throw err;
      const cache = cacheOf(job.settings.cacheReplies);
      if (cache) await forget(cache, replyKey(request));   // an unreadable reply is not kept: asked again, it is asked of the model
      let refusal = err as lmcc.Refusal;
      const thought = response.usage.reasoningTokens ?? 0;
      if (refusal.code === "parse-truncated" && thought) {
        refusal = new lmcc.Refusal(refusal.code, `${refusal.hint} (the model spent ${thought} of its tokens thinking first; raise maxTokens)`,
          { fix: refusal.fix, partial: refusal.partial });
      }
      if (attempt >= retries || !(refusal.code.startsWith("parse-") || refusal.code === "format-read-error")) throw refusal;
      if (refusal.code === "parse-truncated") {
        const current = (configOf(job.settings, overrides as never)?.maxTokens) ?? 1024;
        overrides["maxTokens"] = current * 2;
        request = lm15Request(rendered, job.model, configOf(job.settings, overrides as never));
      } else {
        const correction = `Your reply could not be read: ${refusal.hint}. Reply again, in exactly the form the instructions give.`;
        request = { ...request, messages: [...request.messages, response.message, Message.user(correction)] };
        asked = { ...asked, messages: [...(asked["messages"] as unknown[]), Message.toJSON(response.message),
          { role: "user", parts: [{ type: "text", text: correction }] }] };
        requestHash = lmcc.sha256(asked as lmcc.Json);
      }
      job.call.event("retry", { reason: askedAgain(refusal), wait: null });
    }
  }
}

async function runTool(tools: readonly Tool[], call: { name: string; input?: unknown }, errors: string): Promise<string> {
  const tool = tools.find((t) => t.name === call.name);
  if (!tool) {
    if (errors === "raise") throw new Error(`the model called unknown tool ${JSON.stringify(call.name)}`);
    return `error: there is no tool named ${JSON.stringify(call.name)}`;
  }
  let out: unknown;
  try {
    out = await tool.run(call.input ?? {});
  } catch (err) {
    if (errors === "raise") throw err;
    const e = err as { name?: unknown; message?: unknown } | null | undefined;      // a tool may throw anything, undefined included
    return `error: ${typeof e?.name === "string" ? e.name : "Error"}: ${typeof e?.message === "string" ? e.message : String(err)}`;
  }
  return typeof out === "string" ? out : writeData(out);     // members in the value's order
}

/** Call the model (and run tools until it answers). */
export async function run(job: Job): Promise<Prediction> {
  const { plan } = job;
  const values = prepareInputs(plan.signature, job.inputs);
  if (job.tools.length) setOwn(values, "tools", job.tools.map((t) => ({ name: t.name, description: t.description ?? null, parameters: t.parameters })));
  let turn = plan.turn(values);
  const responses: Response[] = [];
  const toolCalls: Rec[] = [];               // every tool call the model asked for, across steps (outputs.calls)
  const steps = Math.max(1, job.settings.maxSteps);
  for (let i = 0; i < steps; i++) {
    const rendered = plan.render(turn, { turns: [...job.past] });
    const [response, reading] = await complete(job, rendered, responses);
    turn = bridge.step(rendered, response);
    const calls = ((reading.values as Rec)["calls"] ?? []) as Array<{ id: string; name: string; input?: unknown }>;
    toolCalls.push(...(calls as unknown as Rec[]));
    if (!calls.length) {
      turn = turn.finish();
      const outputs: Rec = {};
      for (const [k, v] of entriesOf(turn.outputs ?? {})) if (k !== "calls") setOwn(outputs, k, v);
      const pred = new Prediction(outputs, job.answer, job.call.id, turn, response, responses, reading.repairs);
      if (job.tools.length) Object.defineProperty(pred, "toolCalls", { value: toolCalls, enumerable: false });
      return pred;
    }
    for (const c of calls) {
      const asked = job.call.event("tool_call", { id: c.id, name: c.name, input: c.input ?? {} });
      // a required journal keeps the request before the tool runs (at most its timeout; a cancelled call stops waiting)
      if (await job.call.node?.log.barrier(asked, job.call.signal) === "cancelled") throw new Cancelled();
      stopped(job);
      const output = await runTool(job.tools, c, job.settings.toolErrors);
      job.call.event("tool_result", { id: c.id, name: c.name, output });
      turn = turn.tool(c.id, output);
    }
  }
  throw new StepLimit(`${job.function}: no answer after ${steps} model steps`, turn);
}
