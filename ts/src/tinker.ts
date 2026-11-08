/**
 * A client for Tinker (Thinking Machines' training service), written from the
 * protocol the official Python SDK speaks (tinker 0.39): JSON for most
 * requests; protobuf for `forward_backward`'s request and for the results of
 * training and sampling; work returned as futures collected with
 * `retrieve_future` (HTTP 408 or `{"type": "try_again"}` until ready); one
 * session per client, kept alive by a heartbeat every 10 seconds. Requests on
 * one model are numbered (`seq_id`) and sent in order, so a step's
 * `forward_backward` and `optim_step` land on the same server cycle.
 *
 * ```ts
 * const tk = new Tinker();                                  // TINKER_API_KEY
 * const run = await tk.createModel("Qwen/Qwen3.5-4B", { rank: 32 });
 * const fb = run.forwardBackward(datums, "cross_entropy");  // sent now, awaited later
 * const op = run.optimStep({ learning_rate: 1e-4 });
 * await fb; await op;
 * const sampler = await run.sampler();                      // the current weights
 * await sampler.sample(prompt, { numSamples: 4, temperature: 1 });
 * tk.close();
 * ```
 *
 * There is no official TypeScript SDK; this one is tested against
 * `tools/fake_tinker.py`, itself held to the official SDK.
 */

import { builtin, env } from "./host.ts";
import { parseData, writeData } from "./values.ts";
import { forwardBackwardRequest, readForwardBackwardOutput, readSampleResponse, type WireChunk, type WireDatum, type WireSequence } from "./protowire.ts";
import { type Chunk, type Picture } from "./students.ts";
import { type Datum } from "./training.ts";

type Rec = Record<string, unknown>;

/** The protocol version this client speaks (the official SDK's it was written from). */
export const PROTOCOL = "0.39.0";
const DEFAULT_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod";

/** Tinker refused or failed a request: its HTTP status (0 when none) and what it said. */
export class TinkerError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, detail: string, what = "") {
    super(`Tinker${what ? ` (${what})` : ""}: ${status ? `${status} ` : ""}${detail}`);
    this.name = "TinkerError";
    this.status = status;
    this.detail = detail;
  }
}

/** Adam's settings for one step. */
export interface AdamParams {
  learning_rate: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weight_decay?: number;
  grad_clip_norm?: number;
}

/** What a training step returns: each datum's outputs (`logprobs`: one per position, or rows of K) and the metrics. */
export interface StepOutput {
  readonly outputs: { readonly logprobs: number[] | number[][] }[];
  readonly metrics: Readonly<Record<string, number>>;
}

/** One sampled answer. */
export type Sequence = WireSequence;

/** Options of a sample. */
export interface SampleOptions {
  numSamples?: number;
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  /** Token ids (or strings) that end an answer. */
  stop?: readonly number[] | readonly string[];
  seed?: number;
  /** Return each prompt position's log-probability. */
  promptLogprobs?: boolean;
  /** Rows of K token ids; row i scores prompt position i + 1 (-1: none). */
  targetPromptLogprobs?: readonly (readonly number[])[];
}

/** A sample's result. */
export interface SampleResult {
  readonly sequences: Sequence[];
  /** Per prompt position (NaN at the first, and where none). */
  readonly promptLogprobs: number[] | null;
  /** Rows of K (NaN where the target was -1). */
  readonly targetPromptLogprobs: number[][] | null;
  readonly promptCacheHitTokens: number;
}

/** A checkpoint as Tinker lists it. */
export interface Checkpoint {
  readonly checkpoint_id: string;
  readonly checkpoint_type: "training" | "sampler" | "external";
  readonly tinker_path: string;
  readonly time: string;
  readonly size_bytes?: number | null;
  readonly expires_at?: string | null;
  readonly user_metadata?: Record<string, string> | null;
  readonly base_model?: string | null;
}

/** Options of the client. */
export interface TinkerOptions {
  /** Default TINKER_API_KEY. */
  apiKey?: string;
  /** Default TINKER_BASE_URL, else Tinker's production service. */
  baseUrl?: string;
  /** Kept with the session (the run's name, the plan's fingerprint): Tinker's billing records show it. */
  userMetadata?: Record<string, string>;
  /** How long one HTTP request may take (default 120 s; results are polled, so this bounds one poll). */
  timeoutMs?: number;
  /** How long a 402 (billing not yet seen after a payment) is waited out (default 10 min). */
  billingPauseMs?: number;
  /** Heartbeat period (default 10 s, as the SDK). */
  heartbeatMs?: number;
  /** How long to wait before asking again after a 402 (default 5 s, as the SDK). */
  billingRetryMs?: number;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The base64 of bytes. */
const b64 = (b: Uint8Array) => Buffer.from(b.buffer, b.byteOffset, b.byteLength).toString("base64");

/** A tinker path's training run and checkpoint id: `tinker://<run>/<kind>/<name>`. */
export function parsePath(path: string): { run: string; checkpoint: string } {
  const m = /^tinker:\/\/([^/]+)\/(.+)$/.exec(path);
  if (!m) throw new TypeError(`not a Tinker path: ${path}`);
  return { run: m[1]!, checkpoint: m[2]! };
}

/** Wire chunks for chunks and their pictures. */
export function wireChunks(chunks: readonly Chunk[], pictures: readonly Picture[]): WireChunk[] {
  return chunks.map((c) => {
    if ("text" in c) return { tokens: c.text };
    const p = pictures[c.picture];
    if (!p) throw new RangeError(`chunk names picture ${c.picture}; there are ${pictures.length}`);
    return { image: p.data, format: p.format, expectedTokens: c.tokens };
  });
}

function jsonChunks(chunks: readonly WireChunk[]): Rec[] {
  return chunks.map((c) => ("tokens" in c ? { tokens: [...c.tokens], type: "encoded_text" }
    : { data: b64(c.image), format: c.format, expected_tokens: c.expectedTokens, type: "image" }));
}

/** A training example with its pictures, as the wire takes it. */
export function wireDatum(d: Datum, pictures: readonly Picture[] = []): WireDatum {
  return { chunks: wireChunks(d.modelInput, pictures), inputs: d.lossFnInputs };
}

function wireSize(d: WireDatum): number {
  let n = 0;
  for (const c of d.chunks) n += "tokens" in c ? c.tokens.length * 4 : c.image.length + 16;
  for (const v of Object.values(d.inputs)) n += (v as unknown[]).flat().length * 8;
  return n;
}

/** Tinker: a session, its models and samplers, and the account's checkpoints and spending. */
export class Tinker {
  readonly baseUrl: string;
  readonly #apiKey: string;
  readonly #timeout: number;
  readonly #billingPause: number;
  readonly #meta: Record<string, string>;
  readonly #heartbeatMs: number;
  /** @internal */
  readonly billingRetryMs: number;
  #session: Promise<string> | null = null;
  #heartbeat: ReturnType<typeof setInterval> | null = null;
  #modelSeq = 0;
  #samplerSeq = 0;
  #closed = false;

  constructor(opts: TinkerOptions = {}) {
    const e = env();
    const key = opts.apiKey ?? e["TINKER_API_KEY"];
    if (!key) throw new TinkerError(0, "no API key: set TINKER_API_KEY (a key from the Tinker console, starting with tml-)");
    if (!key.startsWith("tml-") && !key.startsWith("eyJ")) throw new TinkerError(0, "TINKER_API_KEY is not a Tinker key (they start with tml-)");
    this.#apiKey = key;
    this.baseUrl = (opts.baseUrl ?? e["TINKER_BASE_URL"] ?? DEFAULT_URL).replace(/\/+$/, "");
    this.#timeout = opts.timeoutMs ?? 120_000;
    this.#billingPause = opts.billingPauseMs ?? 600_000;
    this.#meta = { ...(opts.userMetadata ?? {}) };
    this.#heartbeatMs = opts.heartbeatMs ?? 10_000;
    this.billingRetryMs = opts.billingRetryMs ?? 5_000;
  }

  /** @internal How long a 402 is waited out. */
  get billingPauseMs(): number {
    return this.#billingPause;
  }

  // ---- HTTP

  /** One HTTP request; JSON in and out unless `body` is bytes (protobuf) or `accept` asks for protobuf. */
  async request(method: string, path: string, opts: { json?: unknown; bytes?: Uint8Array; query?: Record<string, string | number>; accept?: "proto"; what?: string; attempts?: number } = {}):
    Promise<{ status: number; json: unknown; bytes: Uint8Array | null }> {
    const url = new URL(this.baseUrl + path);
    for (const [k, v] of Object.entries(opts.query ?? {})) url.searchParams.set(k, String(v));
    const headers: Record<string, string> = { "x-api-key": this.#apiKey, "user-agent": `functai-ts (tinker protocol ${PROTOCOL})` };
    let body: string | Uint8Array | undefined;
    if (opts.bytes) {
      headers["content-type"] = "application/x-protobuf";
      const zlib = builtin("node:zlib") as { zstdCompressSync?: (b: Uint8Array) => Uint8Array } | null;
      if (zlib?.zstdCompressSync && opts.bytes.length > 4096) {
        body = zlib.zstdCompressSync(opts.bytes);
        headers["content-encoding"] = "zstd";
      } else body = opts.bytes;
    } else if (opts.json !== undefined) {
      headers["content-type"] = "application/json";
      body = writeData(opts.json);
    }
    if (opts.accept === "proto") headers["accept"] = "application/x-protobuf";
    let lastErr: unknown = null;
    const attempts = opts.attempts ?? 6;
    for (let attempt = 0; attempt < attempts; attempt++) {
      let resp: globalThis.Response;
      try {
        resp = await fetch(url, { method, headers, body: body as never, signal: AbortSignal.timeout(this.#timeout) });
      } catch (err) {
        lastErr = err;
        if (attempt + 1 < attempts) await sleep(Math.min(30_000, 500 * 2 ** attempt));
        continue;
      }
      const ctype = resp.headers.get("content-type") ?? "";
      if (ctype.includes("protobuf")) return { status: resp.status, json: null, bytes: new Uint8Array(await resp.arrayBuffer()) };
      const text = await resp.text();
      let json: unknown = null;
      try {
        json = text ? parseData(text) : null;
      } catch {
        json = text;
      }
      if (resp.status >= 500 || resp.status === 429) {                // the service's trouble: again, with backoff
        lastErr = new TinkerError(resp.status, detailOf(json), opts.what ?? path);
        if (attempt + 1 < attempts) await sleep(Math.min(30_000, 500 * 2 ** attempt));
        continue;
      }
      return { status: resp.status, json, bytes: null };
    }
    throw lastErr instanceof TinkerError ? lastErr : new TinkerError(0, `cannot reach ${this.baseUrl}: ${(lastErr as Error)?.message ?? String(lastErr)}`, opts.what);
  }

  /** A JSON request that must succeed (2xx), its JSON. A 402 in the first minutes after paying is waited out. */
  async call(method: string, path: string, json?: unknown, opts: { query?: Record<string, string | number>; what?: string } = {}): Promise<Rec> {
    const started = Date.now();
    for (;;) {
      const r = await this.request(method, path, { json, query: opts.query, what: opts.what });
      if (r.status === 402 && Date.now() - started < this.#billingPause) {
        await sleep(this.billingRetryMs);
        continue;
      }
      if (r.status >= 300) throw new TinkerError(r.status, detailOf(r.json), opts.what ?? path);
      return (r.json ?? {}) as Rec;
    }
  }

  /** A future's result: polled until ready (408 and try_again wait; 402 waits out a billing pause). */
  async result(future: Rec, kind: "json" | "proto", what: string): Promise<{ json: Rec | null; bytes: Uint8Array | null }> {
    const id = future["request_id"] as string;
    const started = Date.now();
    let bad400 = 0;
    for (;;) {
      if (this.#closed) throw new TinkerError(0, "the client was closed", what);
      const r = await this.request("POST", "/api/v1/retrieve_future", { json: { request_id: id, allow_metadata_only: false }, accept: kind === "proto" ? "proto" : undefined, what });
      if (r.bytes) return { json: null, bytes: r.bytes };
      const j = (r.json ?? {}) as Rec;
      if (r.status === 408 || j["type"] === "try_again") continue;
      if (r.status === 402 && Date.now() - started < this.#billingPause) {
        await sleep(this.billingRetryMs);
        continue;
      }
      if (r.status === 400 && bad400++ < 3) continue;                  // the SDK retries a bare 400 (a proxy's) three times
      if (r.status === 410) throw new TinkerError(410, "the request's result expired before it was collected", what);
      if (r.status >= 300) throw new TinkerError(r.status, detailOf(r.json), what);
      if (typeof j["error"] === "string") throw new TinkerError(0, `${j["error"]}${j["category"] ? ` (${String(j["category"])})` : ""}`, what);
      return { json: j, bytes: null };
    }
  }

  // ---- the session

  /** The session's id (made on first use, then kept alive by a heartbeat). */
  session(): Promise<string> {
    if (!this.#session) {
      this.#session = this.call("POST", "/api/v1/create_session", { tags: [], user_metadata: this.#meta, sdk_version: PROTOCOL, type: "create_session" }, { what: "create_session" })
        .then((r) => {
          const id = r["session_id"] as string;
          if (this.#closed) return id;
          // one attempt per beat, never retried: a missed beat waits for the next (a retried one would keep a
          // process alive, its retries' timers outliving the unref'd interval, while the service is unreachable)
          this.#heartbeat = setInterval(() => {
            if (this.#closed) return;
            this.request("POST", "/api/v1/session_heartbeat", { json: { session_id: id, type: "session_heartbeat" }, attempts: 1 }).catch(() => {});
          }, this.#heartbeatMs);
          (this.#heartbeat as { unref?: () => void }).unref?.();
          return id;
        });
      this.#session.catch(() => { this.#session = null; });
    }
    return this.#session;
  }

  /** Stop the heartbeat; nothing is sent after. */
  close(): void {
    this.#closed = true;
    if (this.#heartbeat) clearInterval(this.#heartbeat);
    this.#heartbeat = null;
  }

  // ---- models

  /** A new LoRA adapter on `base` (rank 32 by default). */
  async createModel(base: string, opts: { rank?: number; seed?: number; userMetadata?: Record<string, string> } = {}): Promise<TrainingRun> {
    const session = await this.session();
    const lora: Rec = { rank: opts.rank ?? 32, train_unembed: true, train_mlp: true, train_attn: true };
    if (opts.seed !== undefined) lora["seed"] = opts.seed;
    const body: Rec = { session_id: session, model_seq_id: this.#modelSeq++, base_model: base, lora_config: lora, optimizer_config: { type: "adamw" }, type: "create_model" };
    if (opts.userMetadata) body["user_metadata"] = opts.userMetadata;
    const fut = await this.call("POST", "/api/v1/create_model", body, { what: "create_model" });
    const r = await this.result(fut, "json", "create_model");
    return new TrainingRun(this, (r.json!["model_id"] ?? fut["model_id"]) as string, base);
  }

  /** A training run continued from a saved state (`tinker://…/weights/…`), with its optimizer state (`optimizer: true`) or without. */
  async loadState(path: string, opts: { optimizer?: boolean; base?: string; userMetadata?: Record<string, string> } = {}): Promise<TrainingRun> {
    const session = await this.session();
    const body: Rec = { session_id: session, model_seq_id: this.#modelSeq++, path, optimizer: opts.optimizer ?? true, type: "load_weights" };
    if (opts.userMetadata) body["user_metadata"] = opts.userMetadata;
    const fut = await this.call("POST", "/api/v1/load_weights", body, { what: "load_weights" });
    const r = await this.result(fut, "json", "load_weights");
    const id = (r.json?.["model_id"] ?? fut["model_id"] ?? `${session}:train:${body["model_seq_id"] as number}`) as string;
    return new TrainingRun(this, id, opts.base ?? null);
  }

  /** A sampler: of a saved sampler checkpoint (`tinker://…/sampler_weights/…`), or of a base model's own weights. */
  async sampler(of: { path: string } | { base: string }): Promise<Sampler> {
    const session = await this.session();
    const body: Rec = { session_id: session, sampling_session_seq_id: this.nextSamplerSeq(), type: "create_sampling_session" };
    if ("path" in of) body["model_path"] = of.path;
    else body["base_model"] = of.base;
    const r = await this.call("POST", "/api/v1/create_sampling_session", body, { what: "create_sampling_session" });
    return new Sampler(this, r["sampling_session_id"] as string);
  }

  /** @internal */
  nextSamplerSeq(): number {
    return this.#samplerSeq++;
  }

  // ---- the account

  /** The models Tinker trains (`get_server_capabilities`). */
  async models(): Promise<string[]> {
    const r = await this.call("GET", "/api/v1/get_server_capabilities", undefined, { what: "capabilities" });
    return ((r["supported_models"] ?? []) as Rec[]).map((m) => m["model_name"] as string);
  }

  /** Every checkpoint of the account, newest first. */
  async checkpoints(): Promise<Checkpoint[]> {
    const out: Checkpoint[] = [];
    for (let offset = 0; ; offset += 100) {
      const r = await this.call("GET", "/api/v1/checkpoints", undefined, { query: { limit: 100, offset }, what: "checkpoints" });
      const page = (r["checkpoints"] ?? []) as Checkpoint[];
      out.push(...page);
      const cur = r["cursor"] as Rec | undefined;
      if (!page.length || !cur || offset + page.length >= Number(cur["total_count"] ?? 0)) return out;
    }
  }

  /** Delete a checkpoint by its path. */
  async deleteCheckpoint(path: string): Promise<void> {
    const { run, checkpoint } = parsePath(path);
    await this.call("DELETE", `/api/v1/training_runs/${run}/checkpoints/${checkpoint}`, undefined, { what: "delete checkpoint" });
  }

  /** Make a checkpoint expire after `seconds` (null: keep it). */
  async setTtl(path: string, seconds: number | null): Promise<void> {
    const { run, checkpoint } = parsePath(path);
    await this.call("PUT", `/api/v1/training_runs/${run}/checkpoints/${checkpoint}/ttl`, { ttl_seconds: seconds }, { what: "checkpoint ttl" });
  }

  /** Checkpoint storage now: gigabytes, and what it costs a month (null where Tinker does not say). */
  async storage(): Promise<{ gigabytes: number; checkpoints: number; monthly: number | null; ratePerGbMonth: number | null }> {
    const r = await this.call("GET", "/api/v1/billing/usage/checkpoints/current", undefined, { what: "storage" });
    const rows = (r["data"] ?? []) as Rec[];
    const monthly = rows.some((x) => typeof x["estimated_monthly_cost_usd"] === "number") ? rows.reduce((s, x) => s + Number(x["estimated_monthly_cost_usd"] ?? 0), 0) : null;
    return { gigabytes: rows.reduce((s, x) => s + Number(x["size_gigabytes"] ?? 0), 0), checkpoints: rows.reduce((s, x) => s + Number(x["checkpoint_count"] ?? 0), 0),
      monthly, ratePerGbMonth: (r["effective_rate_usd_per_gigabyte_month"] as number | null | undefined) ?? null };
  }

  /** What was billed between two times (hour buckets), by session: dollars and tokens. */
  async billed(from: Date, to: Date = new Date()): Promise<{ bySession: Record<string, number>; total: number; through: string | null; events: Rec[] }> {
    const hour = (d: Date, up: boolean) => {
      const x = new Date(d);
      x.setUTCMinutes(0, 0, 0);
      if (up && x.getTime() < d.getTime()) x.setUTCHours(x.getUTCHours() + 1);
      return x.toISOString();
    };
    const r = await this.call("GET", "/api/v1/billing/usage/events", undefined, { query: { starting_on: hour(from, false), ending_before: hour(to, true) }, what: "billing" });
    const events = (r["data"] ?? []) as Rec[];
    const bySession: Record<string, number> = {};
    for (const e of events) {
      const s = String(e["session_id"] ?? "");
      bySession[s] = (bySession[s] ?? 0) + Number(e["estimated_cost_usd"] ?? 0);
    }
    return { bySession, total: Object.values(bySession).reduce((a, b) => a + b, 0), through: (r["cost_data_through"] as string | undefined) ?? null, events };
  }
}

function detailOf(json: unknown): string {
  if (json && typeof json === "object") {
    const d = (json as Rec)["detail"] ?? (json as Rec)["error"] ?? (json as Rec)["message"];
    if (typeof d === "string") return d;
    if (d !== undefined) return writeData(d);
  }
  return typeof json === "string" ? json.slice(0, 500) : json === undefined ? "" : writeData(json).slice(0, 500);
}

/** The largest training request Tinker takes in one piece (the SDK's defaults). */
const MAX_DATUMS = 1024;
const MAX_BYTES = 5_000_000;

/** A model being trained: its requests numbered and sent in order. */
export class TrainingRun {
  readonly tinker: Tinker;
  readonly modelId: string;
  readonly base: string | null;
  #seq = 0;
  #queue: Promise<unknown> = Promise.resolve();

  constructor(tinker: Tinker, modelId: string, base: string | null) {
    this.tinker = tinker;
    this.modelId = modelId;
    this.base = base;
  }

  /** Send in order (each request's seq_id after the last's); returns the future, sent. */
  #send<T>(make: (seq: number) => Promise<T>): Promise<T> {
    const p = this.#queue.then(() => make(++this.#seq));
    this.#queue = p.catch(() => {});
    return p;
  }

  /**
   * One forward and backward pass (gradients add up until `optimStep`).
   * Large batches go in several requests, as the SDK sends them. `forwardOnly`:
   * log-probabilities only (billed as training).
   */
  async forwardBackward(data: readonly WireDatum[], lossFn: string, opts: { config?: Record<string, number | string>; forwardOnly?: boolean } = {}): Promise<StepOutput> {
    const pieces: WireDatum[][] = [];
    let cur: WireDatum[] = [];
    let size = 0;
    for (const d of data) {
      const s = wireSize(d);
      if (cur.length && (cur.length >= MAX_DATUMS || size + s > MAX_BYTES)) {
        pieces.push(cur);
        cur = [];
        size = 0;
      }
      cur.push(d);
      size += s;
    }
    if (cur.length) pieces.push(cur);
    const futures = pieces.map((piece) => this.#send((seq) => this.#fwdbwd(piece, lossFn, seq, opts)));
    const results = await Promise.all((await Promise.all(futures)).map((f) => this.tinker.result(f, "proto", opts.forwardOnly ? "forward" : "forward_backward")));
    const outputs: { logprobs: number[] | number[][] }[] = [];
    const metrics: Record<string, number> = {};
    for (const r of results) {
      const o = readForwardBackwardOutput(r.bytes!);
      for (const d of o.outputs) {
        const lp = d["logprobs"] ?? { data: [], shape: [0] };
        const k = lp.shape.length === 2 ? lp.shape[1]! : 0;
        outputs.push({ logprobs: k ? Array.from({ length: lp.shape[0]! }, (_, i) => lp.data.slice(i * k, (i + 1) * k)) : lp.data });
      }
      for (const [key, v] of Object.entries(o.metrics)) metrics[key] = (metrics[key] ?? 0) + v;
    }
    return { outputs, metrics };
  }

  async #fwdbwd(piece: readonly WireDatum[], lossFn: string, seq: number, opts: { config?: Record<string, number | string>; forwardOnly?: boolean }): Promise<Rec> {
    const bytes = forwardBackwardRequest({ modelId: this.modelId, seqId: seq, lossFn, data: piece, config: opts.config, forwardOnly: opts.forwardOnly });
    const started = Date.now();
    for (;;) {
      const r = await this.tinker.request("POST", "/api/v1/forward_backward", { bytes, what: lossFn });
      if (r.status === 402 && Date.now() - started < this.tinker.billingPauseMs) {   // a payment Tinker has not seen yet
        await sleep(this.tinker.billingRetryMs);
        continue;
      }
      if (r.status >= 300) throw new TinkerError(r.status, detailOf(r.json), lossFn);
      return r.json as Rec;
    }
  }

  /** One optimizer step (Adam). */
  async optimStep(adam: AdamParams): Promise<Rec> {
    const params = { learning_rate: adam.learning_rate, beta1: adam.beta1 ?? 0.9, beta2: adam.beta2 ?? 0.95, eps: adam.eps ?? 1e-12,
      weight_decay: adam.weight_decay ?? 0, grad_clip_norm: adam.grad_clip_norm ?? 0 };
    const fut = await this.#send((seq) => this.tinker.call("POST", "/api/v1/optim_step", { model_id: this.modelId, seq_id: seq, type: "optim_step", adam_params: params }, { what: "optim_step" }));
    return (await this.tinker.result(fut, "json", "optim_step")).json ?? {};
  }

  /** Save the training state (weights and optimizer), to continue from: its `tinker://…/weights/<name>` path. */
  async saveState(name: string, opts: { ttlSeconds?: number; userMetadata?: Record<string, string> } = {}): Promise<string> {
    const fut = await this.#send((seq) => this.tinker.call("POST", "/api/v1/save_weights", {
      model_id: this.modelId, path: name, seq_id: seq, ...(opts.ttlSeconds ? { ttl_seconds: opts.ttlSeconds } : {}), overwrite: false,
      ...(opts.userMetadata ? { user_metadata: opts.userMetadata } : {}), type: "save_weights" }, { what: "save_weights" }));
    return (await this.tinker.result(fut, "json", "save_weights")).json!["path"] as string;
  }

  /** Save the weights for sampling: its `tinker://…/sampler_weights/<name>` path. */
  async saveForSampler(name: string, opts: { ttlSeconds?: number; userMetadata?: Record<string, string> } = {}): Promise<string> {
    const fut = await this.#send((seq) => this.tinker.call("POST", "/api/v1/save_weights_for_sampler", {
      model_id: this.modelId, path: name, kind: "snapshot", seq_id: seq, ...(opts.ttlSeconds ? { ttl_seconds: opts.ttlSeconds } : {}),
      ...(opts.userMetadata ? { user_metadata: opts.userMetadata } : {}), type: "save_weights_for_sampler" }, { what: "save_weights_for_sampler" }));
    return (await this.tinker.result(fut, "json", "save_weights_for_sampler")).json!["path"] as string;
  }

  /** A sampler of the current weights, kept nowhere (Tinker deletes it once newer ones are published): for rollouts and checks. */
  async sampler(): Promise<Sampler> {
    const fut = await this.#send((seq) => this.tinker.call("POST", "/api/v1/save_weights_for_sampler", {
      model_id: this.modelId, sampling_session_seq_id: this.tinker.nextSamplerSeq(), kind: "snapshot", seq_id: seq, type: "save_weights_for_sampler" }, { what: "sampler" }));
    const r = await this.tinker.result(fut, "json", "sampler");
    return new Sampler(this.tinker, r.json!["sampling_session_id"] as string);
  }
}

/** A sampler: answers prompts with the weights it was made from. */
export class Sampler {
  readonly tinker: Tinker;
  readonly id: string;
  #seq = 0;

  constructor(tinker: Tinker, id: string) {
    this.tinker = tinker;
    this.id = id;
  }

  /** Answers for a prompt (`numSamples` of them, one request; the prompt billed once). */
  async sample(prompt: readonly WireChunk[], opts: SampleOptions = {}): Promise<SampleResult> {
    const params: Rec = { max_tokens: opts.maxTokens ?? 1024, temperature: opts.temperature ?? 1, top_k: opts.topK ?? -1, top_p: opts.topP ?? 1 };
    if (opts.stop?.length) params["stop"] = [...opts.stop];
    if (opts.seed !== undefined) params["seed"] = opts.seed;
    const body: Rec = { num_samples: opts.numSamples ?? 1, prompt: { chunks: jsonChunks(prompt) }, sampling_params: params, sampling_session_id: this.id,
      seq_id: this.#seq++, prompt_logprobs: Boolean(opts.promptLogprobs), topk_prompt_logprobs: 0, topk_sample_logprobs: 0, prompt_alt_tokens_k: 0,
      record_stability_info: false, type: "sample" };
    if (opts.targetPromptLogprobs) {
      // sparse (CSR), -1 entries left out: what Thinking Machines' own decision recipe sends
      const rows = opts.targetPromptLogprobs;
      const k = rows[0]?.length ?? 0;
      const data: number[] = [];
      const cols: number[] = [];
      const crow: number[] = [0];
      for (const row of rows) {
        row.forEach((t, j) => {
          if (t >= 0) {
            data.push(t);
            cols.push(j);
          }
        });
        crow.push(data.length);
      }
      body["target_prompt_logprobs"] = { data, dtype: "int64", shape: [rows.length, k], sparse_crow_indices: crow, sparse_col_indices: cols };
    }
    const fut = await this.tinker.call("POST", "/api/v1/asample", body, { what: "sample" });
    const r = await this.tinker.result(fut, "proto", "sample");
    const s = readSampleResponse(r.bytes!);
    let target: number[][] | null = null;
    if (s.targetPromptLogprobs) {
      const [n, k] = [s.targetPromptLogprobs.shape[0]!, s.targetPromptLogprobs.shape[1] ?? 1];
      target = Array.from({ length: n }, (_, i) => s.targetPromptLogprobs!.data.slice(i * k, (i + 1) * k));
    }
    return { sequences: s.sequences, promptLogprobs: s.promptLogprobs, targetPromptLogprobs: target, promptCacheHitTokens: s.promptCacheHitTokens };
  }

  /** Each position's log-probability under this sampler's weights (NaN at the first). */
  async logprobs(prompt: readonly WireChunk[]): Promise<number[]> {
    const r = await this.sample(prompt, { maxTokens: 1, promptLogprobs: true, temperature: 1 });
    return r.promptLogprobs ?? [];
  }
}
