/**
 * Baking's language-neutral half (contract/baked.md): what a generative
 * student is trained on and called with, so a model trained by any language,
 * or by any trainer from examples one language wrote, is called correctly by
 * every other.
 *
 * ```ts
 * const table = bakeExamples(summarize, rows);                     // the training conversations
 * await exportExamples("summaries.jsonl", summarize, rows);        // for TRL, Axolotl, Unsloth, a service
 * const student = baked("baked/summarize", { url: "http://localhost:8000/v1" });   // vLLM serving the folder
 * await summarize.using({ lm: student })("...");                   // the function, on the weights
 * ```
 *
 * Training itself is not here: TypeScript writes the examples every trainer
 * reads and runs the weights wherever they are served; Python's
 * `functai.bake` trains them (on this machine, on Tinker, on Prime), or any
 * trainer does from the exported table. Weights trained elsewhere are adopted
 * when their folder's `baked.json` says which function they answer and its
 * signature still has the fingerprint it was baked with.
 */

import * as lmcc from "lmcc";
import { Response, type Request } from "@lm15/lm15";
import type { Call } from "./calllog.ts";
import type { Router } from "./engine.ts";
import { BakeError } from "./errors.ts";
import { builtin } from "./host.ts";
import { REGISTRY, resolveAdapter, templateAdapter } from "./layouts.ts";
import { effective, type Settings } from "./settings.ts";
import * as sig from "./signature.ts";
import { copyData, entriesOf, getOwn, parseData, recordOf, toJson, writeData } from "./values.ts";

type Rec = Record<string, unknown>;

const CAPABILITIES: Rec = { instruct: true };
/** `baked.json`'s format. */
export const BAKED_FORMAT = 2;
/** The examples table's metadata format. */
export const EXAMPLES_FORMAT = 1;

/** `sha256:` of a value's canonical JSON (calls.md, "Canonical JSON"): how fixed and derived inputs are kept. */
export const valueHash = (v: unknown): string => lmcc.sha256(toJson(v)[0] as lmcc.Json);

/** What an AI function is, to baking (an `ai()` function). */
interface Baking {
  readonly name: string;
  readonly definition: sig.Definition;
  readonly settings: Settings;
  readonly tools: readonly unknown[];
  state(): { instructions: string | null };
}

/**
 * One function a generative student answers: the contract between training
 * and calling. Training writes every example through it; a call on the baked
 * model lays out its request through the same entry, read back from `baked.json`.
 */
export interface BakeEntry {
  readonly name: string;
  /** lmcc's fingerprint of the function's full signature. */
  readonly fingerprint: string;
  /** What the student reads: the full signature, minus fixed and derived inputs. */
  readonly signature: lmcc.Signature;
  /** The lmcc adapter artifact, replies written from values. */
  readonly layout: Rec;
  readonly outputs: readonly string[];
  readonly reasoning: boolean;
  /** Input → its value's hash. */
  readonly fixed: Readonly<Record<string, string>>;
  /** Input → `{ from, values: { [hash of a source value]: hash of its value } }`. */
  readonly derived: Readonly<Record<string, Rec>>;
  readonly capabilities: Rec;
}

const leftOut = (e: BakeEntry) => [...Object.keys(e.fixed), ...Object.keys(e.derived)];

function fullSignature(fn: Baking, cot: boolean): lmcc.Signature {
  const s = effective(fn.settings);
  return sig.signature({ ...fn.definition, cot, tools: false, includeName: s.includeFnName !== false }, fn.state().instructions);
}

function reduced(signature: lmcc.Signature, out: readonly string[]): lmcc.Signature {
  if (!out.length) return signature;
  const d = lmcc.signatureToDict(signature) as Rec;
  d["fields"] = (d["fields"] as Rec[]).filter((f) => !(f["direction"] === "input" && out.includes(f["name"] as string)));
  return lmcc.signatureFromDict(d);
}

/** The layout a student is trained and called with: the given one, else the function's own; with replies written from values. */
function studentLayout(fn: Baking, layout: unknown): Rec {
  const s = effective(fn.settings);
  const adapter = layout !== undefined && layout !== null
    ? (Array.isArray(layout) ? templateAdapter(layout as Rec[]) : resolveAdapter(layout))
    : s.template ? templateAdapter(s.template as Rec[]) : resolveAdapter(s.adapter ?? "xml");
  const art = lmcc.dump(adapter, REGISTRY) as Rec;
  art["replay"] = "values";
  return art;
}

/** Options for what a student reads. */
export interface BakeOptions {
  /** A layout other than the function's own (a name, an lmcc adapter, or a template). */
  layout?: unknown;
  /** Keep the reasoning output (with `module: "cot"`): the student learns to reason first. */
  reasoning?: boolean;
  /** Inputs with one value in every row, left out of what the student reads: `{ guide: "Write like…" }`. */
  fixed?: Readonly<Rec>;
  /** Inputs decided by another input the student still reads, left out: `{ guidance: "section" }`. */
  derived?: Readonly<Record<string, string>>;
}

/**
 * What a student of `fn` reads (contract/baked.md, "The examples"): `fn`'s
 * signature without the inputs left out (`fixed`: one value in every row,
 * its hash kept; `derived`: decided by another input, a table of hashes
 * kept), without `reasoning` unless the bake keeps it, laid out in `fn`'s
 * layout with replies written from values, and no worked examples. `rows`
 * are checked against `fixed` and `derived`.
 */
export function bakeEntry(fn: Baking, opts: BakeOptions & { rows?: readonly Rec[] } = {}): BakeEntry {
  if (fn.tools.length) throw new BakeError("bake-rows", `${fn.name} uses tools; training tool-calling students is not supported yet`);
  const s = effective(fn.settings);
  const cot = Boolean(opts.reasoning) && s.module === "cot";
  const full = fullSignature(fn, cot);
  const names = full.fields.filter((f) => f.direction === "input" && f.purpose === "plain").map((f) => f.name);
  const fixed = opts.fixed ?? {};
  const derived = opts.derived ?? {};
  for (const n of [...Object.keys(fixed), ...Object.keys(derived)]) {
    if (!names.includes(n)) throw new BakeError("bake-rows", `${fn.name} has no input ${JSON.stringify(n)} to leave out (its inputs: ${names.join(", ")})`);
  }
  const both = Object.keys(fixed).filter((k) => Object.hasOwn(derived, k));
  if (both.length) throw new BakeError("bake-rows", `${both.sort().join(", ")} cannot be both fixed and derived`);
  for (const [n, src] of Object.entries(derived)) {
    if (!names.includes(src) || Object.hasOwn(fixed, src) || Object.hasOwn(derived, src)) {
      throw new BakeError("bake-rows", `derived: { ${n}: ${JSON.stringify(src)} }: ${JSON.stringify(src)} must be another input the student still reads`);
    }
  }
  const prepared = (vals: Rec) => sig.prepareInputs(full, vals);
  const fixedHashes: Record<string, string> = {};
  for (const [n, v] of Object.entries(fixed)) fixedHashes[n] = valueHash(prepared({ [n]: v })[n]);
  const rows = opts.rows ?? [];
  rows.forEach((row, i) => {
    for (const [n, h] of Object.entries(fixedHashes)) {
      if (Object.hasOwn(row, n) && valueHash(prepared({ [n]: row[n] })[n]) !== h) {
        throw new BakeError("bake-rows", `fixed: { ${n}: … }: row ${i + 1} gives ${n} another value; a fixed input has one value in every row`);
      }
    }
  });
  const table: Record<string, Rec> = {};
  for (const [n, src] of Object.entries(derived)) {
    const values: Record<string, string> = {};
    rows.forEach((row, i) => {
      if (!Object.hasOwn(row, n) || !Object.hasOwn(row, src)) throw new BakeError("bake-rows", `derived: { ${n}: ${JSON.stringify(src)} }: row ${i + 1} lacks ${n} or ${src}`);
      const p = prepared({ [n]: row[n], [src]: row[src] });
      const k = valueHash(p[src]);
      const v = valueHash(p[n]);
      if (Object.hasOwn(values, k) && values[k] !== v) {
        throw new BakeError("bake-rows", `derived: { ${n}: ${JSON.stringify(src)} }: two rows with the same ${src} give ${n} different values, so ${src} does not decide it. Keep ${n} as an input`);
      }
      values[k] = v;
    });
    table[n] = { from: src, values };
  }
  return {
    name: fn.name, fingerprint: lmcc.signatureFingerprint(full), signature: reduced(full, [...Object.keys(fixed), ...Object.keys(derived)]),
    layout: studentLayout(fn, opts.layout), outputs: full.fields.filter((f) => f.direction === "output" && f.purpose !== "tools.calls").map((f) => f.name),
    reasoning: cot, fixed: fixedHashes, derived: table, capabilities: { ...CAPABILITIES },
  };
}

/** @internal The plan a student is laid out with. */
export function studentPlan(e: BakeEntry): lmcc.Plan {
  const adapter = lmcc.load(copyData(e.layout), { registry: REGISTRY });
  return lmcc.bind(adapter, e.signature, e.capabilities, REGISTRY);
}

/**
 * An lm15 request (canonical JSON) as chat-template messages: the system
 * text first, a developer message as a system one, each message's text parts joined.
 */
export function chatMessages(request: Rec): Rec[] {
  const out: Rec[] = [];
  const system = request["system"];
  if (system !== undefined && system !== null && (typeof system !== "string" || system)) {
    out.push({ role: "system", content: typeof system === "string" ? system : (system as Rec[]).map((p) => p["text"]).join("") });
  }
  for (const m of request["messages"] as Rec[]) {
    const role = m["role"] as string;
    if (!["user", "assistant", "system", "developer"].includes(role)) throw new BakeError("bake-rows", `a generative student reads text chats; the request has a ${JSON.stringify(role)} message`);
    const texts = (m["parts"] as Rec[]).map((p) => {
      if (p["type"] !== "text") throw new BakeError("bake-rows", `a generative student reads text; the request carries a ${p["type"]} part`);
      return p["text"] as string;
    });
    out.push({ role: role === "developer" ? "system" : role, content: texts.join("") });
  }
  return out;
}

/**
 * The chat messages of one row's call as the student sees it, and the reply
 * the layout writes for its outputs (null without outputs): a fresh render
 * of the call with no earlier turns, and the assistant message of the same
 * call rendered with that example as its one earlier turn.
 */
export function studentMessages(e: BakeEntry, inputs: Rec, outputs?: Rec | null): [Rec[], string | null] {
  const plan = studentPlan(e);
  const values = sig.prepareInputs(e.signature, recordOf(entriesOf(inputs).filter(([k]) => !leftOut(e).includes(k))));
  const messages = chatMessages(plan.render(plan.turn(values)).request("student") as Rec);
  if (!outputs) return [messages, null];
  const shown = recordOf(e.outputs.filter((k) => Object.hasOwn(outputs, k)).map((k) => [k, toJson(outputs[k])[0]] as const));
  const example = plan.example(values, shown);
  const both = chatMessages(plan.render(plan.turn(values), { turns: [example] }).request("student") as Rec);
  return [messages, both.filter((m) => m["role"] === "assistant").at(-1)!["content"] as string];
}

/** An entry as `baked.json` keeps it. */
export function entryJson(e: BakeEntry): Rec {
  return {
    name: e.name, fingerprint: e.fingerprint, signature: lmcc.signatureToDict(e.signature), layout: copyData(e.layout), outputs: [...e.outputs],
    reasoning: e.reasoning, fixed: { ...e.fixed }, derived: copyData(e.derived), capabilities: { ...e.capabilities },
  };
}

function entryOf(d: Rec): BakeEntry {
  return {
    name: d["name"] as string, fingerprint: d["fingerprint"] as string, signature: lmcc.signatureFromDict(d["signature"]), layout: d["layout"] as Rec,
    outputs: [...((d["outputs"] ?? []) as string[])], reasoning: d["reasoning"] === true, fixed: (d["fixed"] ?? {}) as Record<string, string>,
    derived: (d["derived"] ?? {}) as Record<string, Rec>, capabilities: (d["capabilities"] ?? CAPABILITIES) as Rec,
  };
}

/** One training conversation (contract/baked.md, "The examples table"). */
export interface Example {
  readonly function: string;
  /** The prompt, then `{ role: "assistant", content: <reply> }`: the loss belongs on the reply only. */
  readonly messages: Rec[];
  readonly tag: unknown;
  readonly weight: number;
  readonly row_id: number;
  readonly split: "train" | "validation";
  readonly source: "data";
}

/** Options of `bakeExamples`: what the student reads, and how rows are held out. */
export interface ExamplesOptions extends BakeOptions {
  /** The share of rows held out for validation (default 0.1). */
  validation?: number;
  seed?: number;
  /** A column of the rows: how many times each counts (default 1). */
  weight?: string;
  /** A column of the rows: any label (which teacher wrote it, which source it came from). */
  tag?: string;
}

function shuffled(n: number, seed: number): number[] {
  let a = seed >>> 0;
  const rng = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

/**
 * The training conversations of `fn` on rows with known answers
 * (contract/baked.md, "The examples table"): one per row, `function`,
 * `messages` (the prompt, then the reply: the loss belongs on the reply
 * only), `tag`, `weight`, `row_id`, `split` (a seeded share held out) and
 * `source` (`"data"`). Turning messages into tokens is the student's own
 * chat template's rule, so any trainer can train on them, and a model trained
 * on them is called by `baked()` with the same messages.
 */
export function bakeExamples(fn: Baking, rows: readonly Rec[], opts: ExamplesOptions = {}): Example[] {
  const e = bakeEntry(fn, { ...opts, rows });
  const n = rows.length;
  const held = new Set(shuffled(n, opts.seed ?? 0).slice(0, n > 1 ? Math.min(n - 1, Math.round(n * (opts.validation ?? 0.1))) : 0));
  return rows.map((row, i) => {
    const inputs = recordOf(fn.definition.inputs.filter((f) => getOwn(row, f.name) !== undefined && getOwn(row, f.name) !== null).map((f) => [f.name, row[f.name]] as const));
    for (const [k, v] of Object.entries(opts.fixed ?? {})) if (!Object.hasOwn(inputs, k)) inputs[k] = v;
    const outs = recordOf(e.outputs.filter((k) => getOwn(row, k) !== undefined && getOwn(row, k) !== null).map((k) => [k, row[k]] as const));
    if (!Object.keys(outs).length) throw new BakeError("bake-rows", `row ${i + 1} has no answer for ${e.outputs.join(", ")}: a student learns from rows with known answers`);
    const [messages, reply] = studentMessages(e, inputs, outs);
    return {
      function: e.name, messages: [...messages, { role: "assistant", content: reply }], tag: opts.tag ? (getOwn(row, opts.tag) ?? null) : null,
      weight: opts.weight ? Number(row[opts.weight]) : 1, row_id: i, split: held.has(i) ? "validation" : "train", source: "data",
    };
  });
}

/**
 * `bakeExamples` written as JSON lines (`function`, `messages`, `tag`,
 * `weight`, `row_id`, `split`, `source`), with `<path>.meta.json` beside it
 * (`{"functai_examples": 1, "student": null, "template": null, "functions":
 * [<entry>]}`): what made them, so a model trained on them is adopted by
 * checking its function's fingerprint. Any trainer reads the file (TRL's
 * `SFTTrainer` with `assistant_only_loss`, Axolotl's chat-template datasets,
 * a service's upload). Resolves to the path.
 */
export async function exportExamples(path: string, fn: Baking, rows: readonly Rec[], opts: ExamplesOptions = {}): Promise<string> {
  const fs = builtin("node:fs");
  const nodePath = builtin("node:path");
  if (!fs || !nodePath) throw new Error("exporting examples needs a file system; bakeExamples() gives them as data");
  const table = bakeExamples(fn, rows, opts);
  const e = bakeEntry(fn, { ...opts, rows });
  fs.mkdirSync(nodePath.dirname(nodePath.resolve(path)), { recursive: true });
  fs.writeFileSync(path, table.map((r) => writeData(r) + "\n").join(""));
  fs.writeFileSync(path + ".meta.json", writeData({ functai_examples: EXAMPLES_FORMAT, student: null, template: null, functions: [entryJson(e)] }));
  return path;
}

// ------------------------------------------------------------------ a baked model, used

/**
 * A baked generative model, its weights served by an OpenAI-compatible
 * server (vLLM, SGLang, TGI, llama.cpp) that serves its folder: what
 * `baked()` returns, and what `fn.using({ lm: student })` runs on.
 */
export class BakedModel {
  readonly folder: string;
  readonly meta: Rec;
  readonly entries: ReadonlyMap<string, BakeEntry>;
  readonly url: string;
  readonly model: string;
  readonly apiKey: string | null;
  readonly timeout: number;
  constructor(f: { folder: string; meta: Rec; entries: Map<string, BakeEntry>; url: string; model: string; apiKey: string | null; timeout: number }) {
    this.folder = f.folder;
    this.meta = f.meta;
    this.entries = f.entries;
    this.url = f.url;
    this.model = f.model;
    this.apiKey = f.apiKey;
    this.timeout = f.timeout;
  }

  toString(): string {
    return `BakedModel(${JSON.stringify(this.meta["name"] ?? this.folder)} at ${this.url}: ${[...this.entries.keys()].join(", ")})`;
  }
}

/** Whether a setting's model is a baked student. */
export const isBaked = (lm: unknown): lm is BakedModel => lm instanceof BakedModel;

/**
 * A model baked anywhere (Python's `functai.bake`, or a trainer that wrote
 * `baked.json` format 2), its weights served by an OpenAI-compatible server
 * at `url` (`vllm serve <folder>/model` serves `/v1/chat/completions`).
 * `fn.using({ lm: student })` runs `fn` on it, laid out exactly as it was
 * trained: the student's signature and layout, no worked examples, the chat
 * template the server applies with thinking off. A call is refused when `fn`
 * changed since it was baked (`baked-changed`), or gives a fixed input
 * another value (`baked-fixed`) or a derived one a pair the student never
 * saw (`baked-derived`).
 */
export function baked(folder: string, opts: { url: string; model?: string; apiKey?: string; timeout?: number }): BakedModel {
  const fs = builtin("node:fs");
  const path = builtin("node:path");
  if (!fs || !path) throw new Error("reading a baked folder needs a file system");
  const file = path.join(folder, "baked.json");
  if (!fs.existsSync(file)) throw new BakeError("baked-format", `${folder} has no baked.json`);
  const meta = parseData(fs.readFileSync(file, "utf8")) as Rec;
  if (meta["functai_baked"] !== BAKED_FORMAT) {
    throw new BakeError("baked-format", `${folder} is baked.json format ${JSON.stringify(meta["functai_baked"] ?? null)}; this reader reads format ${BAKED_FORMAT} (a format 1 folder: bake it again)`);
  }
  if (meta["kind"] !== "generative") throw new BakeError("baked-format", `${folder} is a ${meta["kind"]} model; TypeScript runs generative students (a head runs in Python)`);
  const entries = new Map(((meta["functions"] ?? []) as Rec[]).map((d) => [d["name"] as string, entryOf(d)]));
  return new BakedModel({
    folder: path.resolve(folder), meta, entries, url: opts.url.replace(/\/+$/, ""),
    model: opts.model ?? (meta["name"] as string | undefined) ?? path.basename(path.resolve(folder)), apiKey: opts.apiKey ?? null,
    timeout: opts.timeout ?? 120_000,
  });
}

/** The entry a function runs through: by name, or (a model of one function) its only one; refused when the function changed. */
function entryFor(b: BakedModel, fn: Baking): BakeEntry {
  const e = b.entries.get(fn.name) ?? (b.entries.size === 1 ? [...b.entries.values()][0] : undefined);
  if (!e) throw new BakeError("baked-changed", `this baked model answers ${[...b.entries.keys()].join(", ")}, not ${fn.name}`);
  const s = effective(fn.settings);
  if (lmcc.signatureFingerprint(fullSignature(fn, e.reasoning && s.module === "cot")) !== e.fingerprint) {
    throw new BakeError("baked-changed", `${fn.name} has changed since it was baked (its inputs, outputs, types or instruction); bake it again`);
  }
  return e;
}

/** The inputs the student reads, after checking fixed inputs have their baked values and derived ones follow their source. */
function studentInputs(e: BakeEntry, signature: lmcc.Signature, inputs: Rec): Rec {
  if (!leftOut(e).length) return inputs;
  const p = sig.prepareInputs(signature, inputs);
  for (const [n, h] of Object.entries(e.fixed)) {
    if (Object.hasOwn(p, n) && valueHash(p[n]) !== h) {
      throw new BakeError("baked-fixed", `${e.name}: this baked model was trained with ${n} fixed to one value, and this call gives another. The student never learned to read ${n}: call it with the baked value, or bake again with this one`);
    }
  }
  for (const [n, d] of Object.entries(e.derived)) {
    const src = d["from"] as string;
    if (!Object.hasOwn(p, src)) continue;
    const want = getOwn((d["values"] ?? {}) as Record<string, string>, valueHash(p[src]));
    if (want === undefined) throw new BakeError("baked-derived", `${e.name}: this baked model never saw this ${src} in training, so it does not know the ${n} that goes with it`);
    if (Object.hasOwn(p, n) && valueHash(p[n]) !== want) throw new BakeError("baked-derived", `${e.name}: ${n} is not the one the baked model learned for this ${src}; bake again with this pair`);
  }
  return recordOf(entriesOf(inputs).filter(([k]) => !leftOut(e).includes(k)));
}

/** Calls an OpenAI-compatible chat server for a baked student: the request's messages as chat messages, thinking off. */
class BakedRouter implements Router {
  readonly model: BakedModel;
  constructor(model: BakedModel) {
    this.model = model;
  }

  resolve(model: string) {
    return { provider: "functai-baked-lm", model };
  }

  async complete(request: Request, opts: { signal?: AbortSignal } = {}): Promise<Response> {
    const b = this.model;
    const gen = (b.meta["generation"] ?? {}) as Rec;
    const { Request: Req } = await import("@lm15/lm15");
    const body: Rec = { model: b.model, messages: chatMessages(Req.toJSON(request) as Rec), temperature: 0, chat_template_kwargs: { enable_thinking: false } };
    const maxNew = request.config?.maxTokens ?? gen["max_new_tokens"];
    if (maxNew !== undefined && maxNew !== null) body["max_tokens"] = maxNew;
    const signals = [AbortSignal.timeout(b.timeout), ...(opts.signal ? [opts.signal] : [])];
    const resp = await fetch(`${b.url}/chat/completions`, {
      method: "POST", body: writeData(body), signal: AbortSignal.any(signals),
      headers: { "content-type": "application/json", ...(b.apiKey ? { authorization: `Bearer ${b.apiKey}` } : {}) },
    });
    const text = await resp.text();
    if (resp.status >= 400) throw new Error(`the baked model's server at ${b.url} answered ${resp.status}: ${text.slice(0, 300)}`);
    const d = parseData(text) as Rec;
    const choice = (d["choices"] as Rec[])[0]!;
    const usage = (d["usage"] ?? {}) as Rec;
    return Response.fromJSON({
      model: (d["model"] ?? b.model) as string, message: { role: "assistant", parts: [{ type: "text", text: ((choice["message"] as Rec)["content"] ?? "") as string }] },
      finish_reason: choice["finish_reason"] === "length" ? "length" : "stop",
      usage: { input_tokens: usage["prompt_tokens"] ?? null, output_tokens: usage["completion_tokens"] ?? null },
    } as never);
  }
}

/** @internal A call of `fn` on a baked student: its plan, inputs, router and model (fn.ts makes the engine's job of it). */
export function bakedJob(fn: unknown, model: BakedModel, settings: Settings, inputs: Rec, call: Call | null, answer: string) {
  const f = fn as Baking;
  const e = entryFor(model, f);
  const s = effective(f.settings);
  return {
    function: f.name, plan: studentPlan(e), inputs: studentInputs(e, fullSignature(f, e.reasoning && s.module === "cot"), inputs),
    settings: settings as never, router: new BakedRouter(model), model: model.model, tools: [], call: call as Call, answer,
    provider: "functai-baked-lm",
  };
}
