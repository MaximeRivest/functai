/**
 * One call of an AI function: lay it out (lmcc), send it (lm15), read the
 * reply (lmcc), run tools until the model answers (contract/functions.md,
 * "When the reply cannot be read"). No prompt text is written here except
 * the one re-ask sentence the contract gives; every other byte of the request
 * comes from the plan.
 */

import * as lmcc from "lmcc";
import * as bridge from "lmcc/lm15";
import { Delta, Message, RETRYABLE_ERRORS, materializeResponse, responseToEvents, type Request, type Response, type StreamEvent } from "@lm15/lm15";
import type { Call } from "./calllog.ts";
import { Context } from "./host.ts";
import { misfit } from "./shapes.ts";
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

/** What watches a call as it is made (a stream); the engine tells it what happens. */
export interface Watch {
  readonly signal: AbortSignal;
  check(): void;
  onText(call: Call, field: string, text: string): void;
  onThinking(call: Call, text: string): void;
  onToolCall(call: Call, id: string, name: string, input: unknown): void;
  onToolResult(call: Call, id: string, name: string, output: string): void;
  onRetry(call: Call, reason: string, wait: number | null): void;
}

/** The stream watching the calls made in this context, if any. */
export const watching = new Context<Watch>();

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
  readonly watch: Watch | null;
  readonly answer: string;
}

const sleep = (ms: number, signal?: AbortSignal) => new Promise<void>((resolve, reject) => {
  const t = setTimeout(resolve, ms);
  signal?.addEventListener("abort", () => { clearTimeout(t); reject(new Cancelled()); }, { once: true });
});

const retryable = (err: unknown) => RETRYABLE_ERRORS.some((cls) => err instanceof cls);

/** One request: through the router, re-sent after transient errors; streamed when watched. */
async function send(job: Job, request: Request): Promise<Response> {
  const { call, watch } = job;
  const retries = Math.max(0, job.settings.apiRetries);
  for (let attempt = 0; ; attempt++) {
    watch?.check();
    const started = Date.now();
    const t0 = performance.now();
    try {
      let response: Response;
      let first: number | null = null;
      const streamed = watch !== null;
      if (watch && job.router.stream) {
        // The view (lmcc's stream reader) never decides the call: a piece it
        // cannot read stops the view, and the whole reply is read below.
        const events: StreamEvent[] = [];
        const view = job.plan.stream();
        let viewing = true;
        const thinkingIsRead = job.plan.signature.fields.some((f) => f.purpose === "reasoning");
        const show = (batch: readonly { kind: string; field?: string; text?: string }[]) => {
          for (const ev of batch) if (ev.kind === "field_delta" && ev.field !== "calls") watch.onText(call, ev.field!, ev.text!);
        };
        for await (const e of job.router.stream(request, { signal: watch.signal })) {
          if (watch.signal.aborted) throw new Cancelled();
          events.push(e);
          if (e.type !== "delta") continue;
          first ??= (performance.now() - t0) / 1000;
          const d = e.delta as { type: string; text?: string };
          if (d.type === "thinking" && !thinkingIsRead && d.text) watch.onThinking(call, d.text);
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
        response = await job.router.complete(request, watch ? { signal: watch.signal } : undefined);
        if (watch) replay(job, response);
      }
      call.exchange(job.model, request, response, started, (performance.now() - t0) / 1000,
        { streamed: streamed && job.router.stream !== undefined, firstDelta: first });
      return response;
    } catch (err) {
      call.exchange(job.model, request, null, started, (performance.now() - t0) / 1000, { error: err, streamed: watch !== null });
      if (watch?.signal.aborted) throw new Cancelled();
      if (!retryable(err) || attempt >= retries) throw err;
      const after = (err as { retryAfter?: number }).retryAfter;
      const wait = typeof after === "number" && after > 0 ? after : Math.min(30, 2 ** attempt) * (0.5 + Math.random());
      watch?.onRetry(call, `the provider failed (${(err as Error).name}); sending again in ${wait.toFixed(1)} s`, wait);
      await sleep(wait * 1000, watch?.signal);
    }
  }
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
  for (const [field, text] of texts) job.watch!.onText(job.call, field, text);
}

/** The first output value that does not fit its shape, as a parse-value refusal. */
function checkValues(plan: lmcc.Plan, values: Rec): void {
  for (const f of plan.signature.fields) {
    if (f.direction !== "output" || f.purpose !== "plain" || !(f.name in values)) continue;
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
  let request = bridge.request(rendered, { model: job.model, config: configOf(job.settings) });
  const retries = Math.max(0, job.settings.retries);
  for (let attempt = 0; ; attempt++) {
    const response = await send(job, request);
    responses.push(response);
    try {
      const reading = bridge.read(job.plan, response);
      checkValues(job.plan, reading.values as Rec);
      return [response, reading];
    } catch (err) {
      if (!lmcc.isRefusal(err)) throw err;
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
        request = bridge.request(rendered, { model: job.model, config: configOf(job.settings, overrides as never) });
      } else {
        request = {
          ...request, messages: [...request.messages, response.message,
            Message.user(`Your reply could not be read: ${refusal.hint}. Reply again, in exactly the form the instructions give.`)],
        };
      }
      job.watch?.onRetry(job.call, askedAgain(refusal), null);
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
    return `error: ${(err as Error).name ?? "Error"}: ${(err as Error).message ?? String(err)}`;
  }
  return typeof out === "string" ? out : JSON.stringify(out);
}

/** Call the model (and run tools until it answers). */
export async function run(job: Job): Promise<Prediction> {
  const { plan } = job;
  const values = prepareInputs(plan.signature, job.inputs);
  if (job.tools.length) values["tools"] = job.tools.map((t) => ({ name: t.name, description: t.description ?? null, parameters: t.parameters }));
  let turn = plan.turn(values);
  const responses: Response[] = [];
  const steps = Math.max(1, job.settings.maxSteps);
  for (let i = 0; i < steps; i++) {
    const rendered = plan.render(turn, { turns: [...job.past] });
    const [response, reading] = await complete(job, rendered, responses);
    turn = bridge.step(rendered, response);
    const calls = ((reading.values as Rec)["calls"] ?? []) as Array<{ id: string; name: string; input?: unknown }>;
    if (!calls.length) {
      turn = turn.finish();
      const outputs: Rec = {};
      for (const [k, v] of Object.entries(turn.outputs ?? {})) if (k !== "calls") outputs[k] = v;
      return new Prediction(outputs, job.answer, job.call.id, turn, response, responses, reading.repairs);
    }
    for (const c of calls) {
      job.watch?.onToolCall(job.call, c.id, c.name, c.input ?? {});
      const output = await runTool(job.tools, c, job.settings.toolErrors);
      job.watch?.onToolResult(job.call, c.id, c.name, output);
      turn = turn.tool(c.id, output);
    }
  }
  throw new StepLimit(`${job.function}: no answer after ${steps} model steps`, turn);
}
