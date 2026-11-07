/**
 * One call of an AI function: lay it out (lmcc), send it (lm15), read the
 * reply (lmcc), run tools until the model answers (contract/functions.md,
 * "When the reply cannot be read"). No prompt text is written here except
 * the one re-ask sentence the contract gives; every other byte of the request
 * comes from the plan.
 */

import * as lmcc from "lmcc";
import * as replies from "./replies.ts";
import * as history from "./history.ts";
import * as plugins from "./plugins.ts";
import { INVOCATION } from "./calllog.ts";
import { effectsOf, pathOf, type Tool } from "./tools.ts";
import type { Approval } from "./errors.ts";
import { Cancelled } from "./errors.ts";
import { sumUsage } from "./usage.ts";
import * as bridge from "lmcc/lm15";
import { Delta, Message, RETRYABLE_ERRORS, Reasoning, materializeResponse, responseToEvents, type Config, type Request, type Response,
  StreamEvent } from "@lm15/lm15";
import type { Call } from "./calllog.ts";
import { misfit } from "./shapes.ts";
import { copyData, entriesOf, setOwn } from "./values.ts";
import { Delivered, Later, resultOf, resultText, type ToolOutput } from "./results.ts";
import { ConversationError } from "./errors.ts";
import { warnOnce } from "./calllog.ts";
import { configOf, type Settings } from "./settings.ts";
import { prepareInputs } from "./signature.ts";

type Rec = Record<string, unknown>;

export type { Tool } from "./tools.ts";
export { Cancelled } from "./errors.ts";

/** The tool loop reached `maxSteps` without an answer. */
export class StepLimit extends Error {
  readonly turn: lmcc.Turn;
  constructor(message: string, turn: lmcc.Turn) {
    super(message);
    this.turn = turn;
    this.name = "StepLimit";
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
  /**
   * Why the call has no answer, when its settings kept an unreadable reply
   * (`onUnreadable: "record"`): the refusal it would have raised. Else null.
   */
  refusal: lmcc.Refusal | null = null;

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

  /**
   * The probabilities the model measured for its own answers, by output
   * (`{ result: { billing: 0.93, shipping: 0.05 } }`); empty when it
   * measures none: only TypeSafe (`typesafe:jev-latest`) and servers that
   * score tokens do (a model's own words about its confidence are not one).
   */
  probabilities: Record<string, Record<string, number>> = {};
  /** The probability the model gave its own answer, the lowest over the outputs it measured; null when it measured none. */
  get confidence(): number | null {
    const values: number[] = [];
    for (const [field, dist] of Object.entries(this.probabilities)) {
      const ps = Object.values(dist);
      if (!ps.length) continue;
      const v = (this.outputs as Rec)[field];
      const key = typeof v === "string" ? v : typeof v === "boolean" ? String(v) : JSON.stringify(v);
      values.push(Object.hasOwn(dist, key) ? dist[key]! : Math.max(...ps));
    }
    return values.length ? Math.min(...values) : null;
  }

  /** A first model was unsure and another answered: the first one's prediction (`escalateTo`). */
  first: Prediction | null = null;
  get escalated(): boolean {
    return this.first !== null;
  }

  /**
   * Tokens, summed over every reply: `inputTokens`, `outputTokens`, `totalTokens`, …
   * (lm15 2: input includes cached tokens, output includes reasoning). A count one
   * reply did not report is absent: a sum is exact or absent.
   */
  get usage(): Record<string, number> {
    return sumUsage(this.responses.map((r) => {
      const counts: Record<string, number> = {};
      for (const [k, v] of Object.entries((r.usage ?? {}) as Record<string, unknown>)) {
        if (typeof v === "number" && Number.isInteger(v)) counts[k] = v;
      }
      return counts;
    }));
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
  readonly settings: Settings & { retries: number; apiRetries: number; apiMaxWait: number; maxSteps: number; toolErrors: string };
  readonly router: Router;
  readonly model: string;
  readonly tools: readonly Tool[];
  readonly call: Call;
  readonly answer: string;
  /** The plugins around the call, in order (contract/plugins.md). */
  readonly plugins: readonly plugins.Plugin[];
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
  if (job.call.node?.log.replaying) return;                 // a resumed turn replaying a request it made before
  job.call.requests += 1;
  job.call.event("request", { request: job.call.requests, model: job.model });
}

const stopped = (job: Job) => {
  if (job.call.signal?.aborted) throw new Cancelled();
};

/**
 * One request: `hit` when the reply is already known (the reply cache, or a
 * turn being resumed: contract/tools.md), else through the router, re-sent
 * after transient errors; streamed when something watches the call live.
 */
async function send(job: Job, request: Request, requestHash: string | null, hit: Response | null): Promise<Response> {
  const { call } = job;
  const signal = call.signal;
  if (hit) {
    stopped(job);
    requested(job);
    call.exchange(job.model, request, hit, Date.now(), 0, { cached: true, requestHash });
    history.keep({ function: job.function, model: job.model, request, response: hit, cached: true, error: null });
    if (call.watched) replay(job, hit);           // a whole reply: one text piece per field (streaming.md)
    return hit;
  }
  const retries = Math.max(0, job.settings.apiRetries);
  for (let attempt = 0; ; attempt++) {
    stopped(job);
    const started = Date.now();
    const t0 = performance.now();
    requested(job);
    const watched = call.node ? call.node.log.wantsPieces(call.node) : false;
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
          // what no other event shows, as lm15 wrote it (streaming.md, *Pieces*): text and thinking are shown below
          const delta = e.type === "delta" ? e.delta : null;
          const d = delta as { type: string; text?: string } | null;
          if (d?.type !== "text" && d?.type !== "thinking") call.event("piece", { lm15: StreamEvent.toJSON(e) });
          if (!delta || !d) continue;
          first ??= (performance.now() - t0) / 1000;
          if (d.type === "thinking" && !thinkingIsRead && d.text) call.event("thinking", { text: d.text });
          if (viewing) {
            try {
              show(view.feed(Delta.toJSON(delta) as Rec) as never);
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
        if (call.watched) replay(job, response);
      }
      call.exchange(job.model, request, response, started, (performance.now() - t0) / 1000,
        { streamed: watched && job.router.stream !== undefined, firstDelta: first, requestHash });
      history.keep({ function: job.function, model: job.model, request, response, cached: false, error: null });
      return response;
    } catch (err) {
      history.keep({ function: job.function, model: job.model, request, response: null, cached: false, error: `${(err as Error)?.name}: ${(err as Error)?.message}` });
      call.exchange(job.model, request, null, started, (performance.now() - t0) / 1000,
        { error: err, streamed: watched && job.router.stream !== undefined, requestHash });
      if (signal?.aborted) throw new Cancelled();
      if (!retryable(err)) throw err;
      const after = (err as { retryAfter?: number }).retryAfter;
      const asked = typeof after === "number" && after > 0 ? after : null;
      const limit = job.settings.apiMaxWait ?? 60;
      if (asked !== null && asked > limit && err instanceof Error) err.message += `\n${longWaitNote(asked, limit)}`;
      if (attempt >= retries || (asked !== null && asked > limit)) throw err;
      const wait = asked ?? Math.min(30, 2 ** attempt) * (0.5 + Math.random());
      call.event("retry", { reason: `the provider failed (${(err as Error).name}); sending again in ${wait.toFixed(1)} s`, wait });
      await sleep(wait * 1000, signal);
    }
  }
}

/** What a provider asking for a longer wait than `apiMaxWait` is told (functions.md). */
export function longWaitNote(asked: number, limit: number): string {
  const until = new Date(Date.now() + asked * 1000).toISOString().slice(0, 16).replace("T", " ");
  return `the provider asks to wait ${duration(asked)} (until ${until} UTC); functai waits at most apiMaxWait=${limit} s, `
    + `so it did not send again. To wait, set apiMaxWait: ${Math.ceil(asked)}; or run again later `
    + `(with cacheReplies: "disk", the answers already received are kept).`;
}

/** `3 h 7 min`, `12 min`, `45 s`. */
export function duration(seconds: number): string {
  const s = Math.round(seconds), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return m ? `${h} h ${m} min` : `${h} h`;
  if (m) return sec && m < 10 ? `${m} min ${sec} s` : `${m} min`;
  return `${sec} s`;
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

/**
 * The rendered request with the function's own `reasoning` setting in the layout's (contract/functions.md,
 * *Reasoning*): it chooses the effort and the thinking budget, and the summary when it names one; the layout's
 * `medium` and `auto` stay where it says nothing. Without it, or when the layout asks for no reasoning, the
 * request is the one rendered.
 */
export function ownReasoning(rendered: lmcc.RenderResult, config: Config | undefined): lmcc.RenderResult {
  const mine = config?.reasoning;
  const settings = rendered.requestSettings as Rec;
  const theirs = ((settings["config"] ?? {}) as Rec)["reasoning"];
  if (!mine || !theirs || typeof theirs !== "object") return rendered;
  const wanted = Object.fromEntries(Object.entries(Reasoning.toJSON(Reasoning.create(mine)) as Rec).filter(([, v]) => v !== undefined && v !== null));
  let merged: Rec = { ...(theirs as Rec), ...wanted };
  if (merged["effort"] === "off") merged = { effort: "off" };       // lm15 refuses a summary or a budget with off (INV-026)
  return new lmcc.RenderResult(rendered.messages, { ...settings, config: { ...(settings["config"] as Rec), reasoning: merged } },
    rendered.system, rendered.plan, rendered.turn);
}

/** The first output value that does not fit its shape, as a parse-value refusal. */
function checkValues(plan: lmcc.Plan, values: Rec): void {
  for (const f of plan.signature.fields) {
    if (f.direction !== "output" || f.purpose !== "plain" || !Object.hasOwn(values, f.name)) continue;
    const problem = misfit(f.shape as Rec, values[f.name], f.name);
    if (problem) throw new lmcc.Refusal("parse-value", problem);
  }
}

const LMCC_TRUNCATED_ADVICE = "; raise max_tokens or ask for less";

/**
 * A `parse-truncated` refusal that says what happened (functions.md, "When the reply cannot be read"):
 * how much went to thinking, whose limit it was and whether it can be raised, and what lm15 changed in
 * the request. `limit`: the maxTokens this request set, or undefined.
 */
export function cutOff(err: lmcc.Refusal, response: Response, limit: number | undefined): lmcc.Refusal {
  let hint = err.hint.endsWith(LMCC_TRUNCATED_ADVICE) ? err.hint.slice(0, -LMCC_TRUNCATED_ADVICE.length) : err.hint;
  const thought = response.usage?.reasoningTokens ?? 0;
  const total = response.usage?.outputTokens ?? 0;
  if (thought) {
    hint += total >= thought ? `; the model spent ${thought} of its ${total} output tokens thinking` : `; the model spent ${thought} tokens thinking`;
  }
  const notes = response.adaptations ?? [];
  const chosen = notes.find((a) => a.field === "config.max_tokens" && a.action === "defaulted")?.applied;
  if (limit !== undefined) hint += `; raise maxTokens (it was ${limit}) or ask for less`;
  else if (chosen !== undefined) {
    hint += `; no maxTokens was set, and lm15 sent ${String(chosen)}, the most it knows this model to allow: lower the reasoning effort or ask for less`;
  } else hint += "; no maxTokens was set, so the provider used its own maximum: lower the reasoning effort or ask for less";
  const other = notes.filter((a) => a.field !== "config.max_tokens").map((a) => `${a.field} ${a.action}: ${a.reason}`);
  if (other.length) hint += ` (lm15 adapted the request: ${other.join("; ")})`;
  return new lmcc.Refusal(err.code, hint, { fix: err.fix, partial: err.partial });
}

function askedAgain(err: lmcc.Refusal): string {
  if (err.code === "parse-truncated") return "the reply was cut off; asking again with a larger token budget";
  if (err.code === "parse-interrupted") return "the reply ended in an error; sending the same request again";
  return `the reply could not be read (${err.hint}); asking again`;
}

/**
 * One model call; after an unreadable reply, up to `retries` follow-ups that
 * send the reader's hint back. Each attempt: the `request` hooks (the escape
 * hatch), then a reply already known (the turn being resumed recorded it, or
 * the reply cache kept it), else the provider. Only a reply that was read is
 * kept in the cache; a stored turn keeps every reply.
 */
async function complete(job: Job, rendered: lmcc.RenderResult, responses: Response[]): Promise<[Response, lmcc.Reading]> {
  const overrides: Rec = {};
  let request = lm15Request(rendered, job.model, configOf(job.settings));
  // lmcc's hash of the request each exchange sends (calls.md, exchanges): the render's; a re-ask's is the one it follows
  // with the reply's message and the correction appended; a re-send with a larger token budget sends the same lmcc request
  let asked = rendered.request() as Rec;
  let requestHash = lmcc.sha256(asked as lmcc.Json);
  const retries = Math.max(0, job.settings.retries);
  const replicate = Number(job.settings.replicate ?? 0) || 0;
  const run = job.call.turnRun;
  for (let attempt = 0; ; attempt++) {
    // the escape hatch: a plugin may replace the provider request; no one can rebuild it, so its exchange has no request_hash
    const [sent, replaced] = await plugins.requestHook(job.plugins, request, job.call, job.function);
    let hit: Response | null = null;
    if (run) {
      hit = run.recordedReply(replies.replyKey(sent, replicate));
      if (!hit) job.call.node?.log.frontier();               // the turn does something it had not done: shown from here
    }
    const flight = hit ? null : await replies.begin(job.settings.cacheReplies, sent, replicate, job.call.content, job.call.signal);
    let response!: Response;
    try {
      response = await send(job, sent, replaced ? null : requestHash, hit ?? flight?.reply ?? null);
      responses.push(response);
      if (run?.durable) {
        try {
          run.noteReply(replies.replyKey(sent, replicate), response);     // a stored turn keeps every reply
        } catch (err) {
          warnOnce(`note-reply:${(err as Error).name}`, `a reply could not be kept in the conversation (${(err as Error).message})`);
        }
      }
      let reading: lmcc.Reading;
      try {
        reading = bridge.read(job.plan, response);
        checkValues(job.plan, reading.values as Rec);
      } catch (err) {
        if (flight) await flight.drop();                     // a kept reply that no longer reads is forgotten
        throw err;
      }
      if (flight && response.finishReason !== "error") await flight.keep(response);   // only a reply that was read is kept
      return [response, reading];
    } catch (err) {
      if (!lmcc.isRefusal(err)) throw err;
      let refusal = err as lmcc.Refusal;
      // the budget this request set (functions.md: a cut reply is re-sent with twice it, only when one was set)
      const limit = configOf(job.settings, overrides as never)?.maxTokens ?? undefined;   // null: none set
      if (refusal.code === "parse-truncated") refusal = cutOff(refusal, response, limit);
      if (attempt >= retries || refusal.code === "parse-filtered"     // lmcc's: a stopped reply would be stopped again
        || !(refusal.code.startsWith("parse-") || refusal.code === "format-read-error")) throw refusal;
      if (refusal.code === "parse-truncated" && limit === undefined) throw refusal;   // nothing larger to give
      if (refusal.code === "parse-truncated") {
        overrides["maxTokens"] = limit! * 2;
        request = lm15Request(rendered, job.model, configOf(job.settings, overrides as never));
      } else if (refusal.code === "parse-interrupted") {
        // the reply ended in an error: the same request again, no correction, no larger budget (functions.md)
      } else {
        const correction = `Your reply could not be read: ${refusal.hint}. Reply again, in exactly the form the instructions give.`;
        request = { ...request, messages: [...request.messages, response.message, Message.user(correction)] };
        asked = { ...asked, messages: [...(asked["messages"] as unknown[]), Message.toJSON(response.message),
          { role: "user", parts: [{ type: "text", text: correction }] }] };
        requestHash = lmcc.sha256(asked as lmcc.Json);
      }
      job.call.event("retry", { reason: askedAgain(refusal), wait: null });
    } finally {
      await flight?.end();
    }
  }
}

/** A promise, or Cancelled once the signal aborts (whichever comes first): a stopped call never waits on code that does not watch it. */
function untilAborted<T>(p: PromiseLike<T> | T, signal: AbortSignal | undefined): Promise<T> {
  if (!signal) return Promise.resolve(p);
  if (signal.aborted) return Promise.reject(new Cancelled());
  return new Promise<T>((resolve, reject) => {
    const stop = () => reject(new Cancelled());
    signal.addEventListener("abort", stop, { once: true });
    Promise.resolve(p).then((v) => { signal.removeEventListener("abort", stop); resolve(v); },
      (e) => { signal.removeEventListener("abort", stop); reject(e); });
  });
}

async function runTool(tools: readonly Tool[], call: { name: string; input?: unknown }, errors: string, signal?: AbortSignal): Promise<ToolOutput | Later> {
  const tool = tools.find((t) => t.name === call.name);
  if (!tool) {
    if (errors === "raise") throw new Error(`the model called unknown tool ${JSON.stringify(call.name)}`);
    return `error: there is no tool named ${JSON.stringify(call.name)}`;
  }
  let out: unknown;
  try {
    out = await untilAborted(tool.run(call.input ?? {}), signal);
  } catch (err) {
    if (errors === "raise" || err instanceof Cancelled || (err as { code?: unknown })?.code === "turn-waiting") throw err;
    const e = err as { name?: unknown; message?: unknown } | null | undefined;      // a tool may throw anything, undefined included
    return `error: ${typeof e?.name === "string" ? e.name : "Error"}: ${typeof e?.message === "string" ? e.message : String(err)}`;
  }
  if (out instanceof Later) return out;
  return resultOf(out);                                      // text, a value as JSON, or pictures and other parts
}

/** The `tool_result` event: `output` the result's text, `parts` the whole result when it holds more than text (streaming.md). */
function showResult(call: Call, id: string, name: string, output: ToolOutput, invocation: number): void {
  if (typeof output === "string") call.event("tool_result", { id, name, output, invocation });
  else call.event("tool_result", { id, name, output: resultText(output), parts: copyData(output), invocation });
}

/** A tool call whose result comes later (`later(key)`). */
class Pending {
  readonly key: string;
  readonly approval: Approval;
  readonly input: unknown;
  readonly asked: { id: string; name: string };
  constructor(key: string, approval: Approval, input: unknown, asked: { id: string; name: string }) {
    this.key = key;
    this.approval = approval;
    this.input = input;
    this.asked = asked;
  }
}

/** A result that came later, as the model is shown it: given to the `toolResult` hooks, then kept as the tool's result. */
async function delivered(job: Job, approval: Approval, input: unknown, output: ToolOutput): Promise<ToolOutput> {
  const current = job.call;
  current.node?.log.frontier();
  const shown = await plugins.toolResult(job.plugins, current, approval, input, output);
  current.turnRun?.toolDone(current, approval, shown);
  return shown;
}

/**
 * The step's results, with those that come later: each one already delivered
 * is used now; when one is not, the turn stops and waits for them (every
 * other tool call of the step ran, and is kept).
 */
async function resultsLater(job: Job, ran: readonly (ToolOutput | Pending)[]): Promise<ToolOutput[]> {
  const run = job.call.turnRun!;
  const pending = ran.filter((o): o is Pending => o instanceof Pending);
  const got = await run.deliveries(pending.map((p) => p.key));
  const missing = pending.filter((p) => !got.has(p.key));
  if (missing.length) {
    run.waitLater(missing.map((p) => ({ site: job.call.path, invocation: p.approval.invocation, id: p.asked.id, name: p.asked.name,
      key: p.key, path: p.approval.path })));
  }
  const out: ToolOutput[] = [];
  for (const o of ran) {
    if (!(o instanceof Pending)) {
      out.push(o);
      continue;
    }
    const shown = await delivered(job, o.approval, o.input, got.get(o.key)!);
    showResult(job.call, o.asked.id, o.asked.name, shown, o.approval.invocation);
    out.push(shown);
  }
  return out;
}

/**
 * One tool call of the call in progress (contract/tools.md, plugins.md):
 * numbered (its invocation), shown, given to the `toolCall` hooks (which may
 * change its input, block it, or ask a person: `approve` is one of them),
 * kept before and after it runs when it changes things (a stored turn, a
 * required journal), run, its result given to the `toolResult` hooks, and
 * shown.
 */
async function oneTool(job: Job, asked: { id: string; name: string; input?: unknown }): Promise<ToolOutput | Pending> {
  const current = job.call;
  const n = ++current.invocations;
  const tool = job.tools.find((t) => t.name === asked.name);
  const effects = tool ? effectsOf(tool) : "reads";            // an unknown tool runs nothing
  const made = current.event("tool_call", { id: asked.id, name: asked.name, input: asked.input ?? {}, invocation: n });
  const approval: Approval = {
    call: current.id, invocation: n, id: asked.id, name: asked.name, input: asked.input ?? {}, effects,
    path: pathOf(current, asked.name), site: current.path, plugin: "approval",
  };
  const [input, refused] = await plugins.toolCall(job.plugins, current, approval, job.settings as unknown as Rec);
  let output: ToolOutput;
  if (refused !== null) output = refused;
  else {
    const run = current.turnRun;
    const known = run ? run.recordedTool(current, approval) : null;
    if (known instanceof Delivered) output = await delivered(job, approval, input, known.output);   // a result that came later
    else if (known !== null) output = known;                   // a turn resumed: this tool ran before; its result is kept
    else {
      current.node?.log.frontier();
      if (effects !== "reads") {
        // a required journal keeps the request before a tool that changes things runs (at most its timeout)
        if (await current.node?.log.barrier(made, current.signal) === "cancelled") throw new Cancelled();
        run?.toolStarted(current, { ...approval, input });
      }
      stopped(job);
      const got = await INVOCATION.run(n, () => runTool(job.tools, { name: asked.name, input }, job.settings.toolErrors, current.signal));
      if (got instanceof Later) {
        if (!run) {
          throw new ConversationError("later-unavailable", `${approval.path} returned later(${JSON.stringify(got.key)}), and its result can only be waited for in a conversation's turn (fn.conversation(...)): this call runs in none. Whatever the tool started may be running.`);
        }
        await run.toolLater(current, { ...approval, input }, got.key);
        current.event("later", { id: asked.id, invocation: n, name: asked.name, key: got.key });
        return new Pending(got.key, approval, input, asked);
      }
      // the result as the model is shown it; a turn resumed later reuses it, hooks and all
      output = await plugins.toolResult(job.plugins, current, approval, input, got);
      run?.toolDone(current, approval, output);
    }
  }
  showResult(current, asked.id, asked.name, output, n);
  return output;
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
  const steering = job.tools.length
    ? new plugins.Steering(job.call, job.function, steps, job.tools.map((t) => t.name), typeof job.settings.lm === "string" ? job.settings.lm : null, job.plugins)
    : null;
  for (let i = 0; i < steps; i++) {
    if (i > 0 && steering) await steering.before(i + 1);     // the step hook: tools offered, reminders, results shown otherwise
    const rendered = ownReasoning(plan.render(steering ? steering.view(turn) : turn, { turns: [...job.past] }), configOf(job.settings));
    let got: [Response, lmcc.Reading];
    try {
      got = await complete(job, rendered, responses);
    } catch (err) {
      if (job.settings.onUnreadable !== "record" || !lmcc.isRefusal(err) || !(err as lmcc.Refusal).code.startsWith("parse-")
        || !responses.length) throw err;
      // kept as it came, with no values (functions.md, "Keeping an unreadable reply"): the call ends well with no answer
      const refusal = err as lmcc.Refusal;
      const last = responses[responses.length - 1]!;
      turn = turn.withStep(new lmcc.ModelStep({}, Message.toJSON(last.message) as never, lmcc.sha256(rendered.request() as lmcc.Json),
        plan.callsField)).finish();
      turn = turn.withMeta({ ...turn.meta, refusal: refusal.describe() as unknown as Rec });
      const pred = new Prediction({}, job.answer, job.call.id, turn, last, responses);
      pred.refusal = refusal;
      return pred;
    }
    const [response, reading] = got;
    // the reply as the rendered turn's next model step; the call's turn keeps every result as its tool gave it (plugins.md, "The step hook")
    const stepped = bridge.step(rendered, response);
    turn = steering ? turn.withStep(stepped.steps[stepped.steps.length - 1]!) : stepped;
    const calls = ((reading.values as Rec)["calls"] ?? []) as Array<{ id: string; name: string; input?: unknown }>;
    toolCalls.push(...(calls as unknown as Rec[]));
    if (!calls.length) {
      turn = turn.finish();
      const outputs: Rec = {};
      for (const [k, v] of entriesOf(turn.outputs ?? {})) if (k !== "calls") setOwn(outputs, k, v);
      const pred = new Prediction(outputs, job.answer, job.call.id, turn, response, responses, reading.repairs);
      pred.probabilities = copyData((reading as { probabilities?: Record<string, Record<string, number>> }).probabilities ?? {});
      if (job.tools.length) Object.defineProperty(pred, "toolCalls", { value: toolCalls, enumerable: false });
      return pred;
    }
    const first = job.call.invocations + 1;
    let ran: (ToolOutput | Pending)[] = [];
    for (const c of calls) {
      stopped(job);
      ran.push(await oneTool(job, c));
    }
    if (ran.some((o) => o instanceof Pending)) ran = await resultsLater(job, ran);
    calls.forEach((c, k) => {
      const output = ran[k] as ToolOutput;
      turn = turn.tool(c.id, output as never);
      steering?.ran(first + k, c.name, c.input ?? {}, output);
    });
  }
  throw new StepLimit(`${job.function}: no answer after ${steps} model steps`, turn);
}
