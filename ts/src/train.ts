/**
 * Training a student of an AI function on Tinker, from TypeScript
 * (design/21, contract/training.md): the plan and its price before anything
 * is spent, a run folder that resumes, five methods, checks by the
 * function's own criteria while it trains, and a model that answers calls on
 * Tinker exactly as it was trained.
 *
 * ```ts
 * const s1 = await bake(scanpath, rows, { student: "Qwen/Qwen3.5-4B", criteria: [match], heldOutBy: "screen" });
 * const s2 = await bake(scanpath, rows, { method: "rl", start: s1, reward: match });
 * await scanpath.using({ lm: s2 })({ screen, width, height, target });
 * ```
 *
 * Methods: `sft` (imitate the rows' answers, or a teacher's), `rl` (a reward:
 * a criterion, or a group reward such as `rewards.energy`), `prefer` (pairs of
 * a better and a worse answer: DPO), `distill` (on-policy: a teacher scores
 * every token the student writes), `decision` (one forward pass gives a
 * probability for each allowed answer).
 */

import * as lmcc from "lmcc";
import * as bridge from "lmcc/lm15";
import { Response, type Request } from "@lm15/lm15";
import { BakedModel, BAKED_FORMAT, bakeEntry, entryJson, entryOf, studentPlan, valueHash, type BakeEntry, type BakeOptions } from "./bake.ts";
import type { Router } from "./engine.ts";
import { BakeError } from "./errors.ts";
import { evaluate } from "./evaluate.ts";
import type { AIFunction } from "./fn.ts";
import { builtin, env } from "./host.ts";
import { checkInputs } from "./interface.ts";
import * as m from "./measures.ts";
import * as sig from "./signature.ts";
import { chunksLength, effortOf, Student, studentConversation, type Chunk, type Picture, type StudentOptions } from "./students.ts";
import { Sampler, Tinker, wireChunks, wireDatum, type TinkerOptions, type TrainingRun } from "./tinker.ts";
import * as T from "./training.ts";
import { parseData, parseForeign, toJson, writeData } from "./values.ts";

type Rec = Record<string, unknown>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Fn = AIFunction<any, any, any>;

// ------------------------------------------------------------------ rewards

/** A reward on a whole group of answers to one case: one value per answer (null answers could not be read). */
export interface GroupReward {
  readonly name: string;
  readonly group: true;
  /** Rewards for the answers, given the case's references (the right answers of every row with these inputs). */
  rewards(answers: readonly (unknown | null)[], references: readonly unknown[], inputs: Rec): { rewards: number[]; score: number };
  /** Higher is better for the score it reports (false: a loss, like the energy score). */
  readonly higherIsBetter: boolean;
}

/** A reward: a criterion with a judge (each answer judged), a function of one answer, or a group reward. */
export type Reward = m.Criterion | ((args: m.JudgeArgs & { references: unknown[] }) => unknown) | GroupReward;

export const rewards = {
  /**
   * The energy score (contract/training.md §8): a group of answers rewarded
   * for being spread the way the references are spread. `distance(a, b)` is
   * the task's distance between two answers, in [0, 1].
   */
  energy(distance: (a: unknown, b: unknown) => number, name = "energy"): GroupReward {
    return {
      name, group: true, higherIsBetter: false,
      rewards(answers, references) {
        const readable = answers.map((a) => a !== null && a !== undefined);
        const dRef = answers.map((a) => (a === null || a === undefined ? [] : references.map((y) => distance(a, y))));
        const dPair = answers.map((a, i) => answers.map((b, j) => (i === j || a === null || a === undefined || b === null || b === undefined ? 0 : distance(a, b))));
        if (!references.length) throw new BakeError("bake-rows", "the energy score needs each case's references (rows with a known answer)");
        return T.energy(readable, dRef, dPair);
      },
    };
  },
};

const isGroup = (r: unknown): r is GroupReward => typeof r === "object" && r !== null && (r as GroupReward).group === true;

// ------------------------------------------------------------------ options

export type Method = "sft" | "rl" | "prefer" | "distill" | "decision";

/** One pair for `prefer`: a case's inputs, its better and its worse answer (the function's outputs). */
export interface PreferencePair {
  readonly inputs: Rec;
  readonly chosen: Rec;
  readonly rejected: Rec;
}

export interface TrainOptions extends BakeOptions {
  /** Tinker's id of the student (`Qwen/Qwen3.5-4B`, `thinkingmachines/Inkling-2-Small`); a `start` model's by default. */
  student?: string;
  /** Where it trains: Tinker (the only service TypeScript trains on). */
  where?: "tinker";
  method?: Method;
  /** A column of the rows: how many times each counts (default 1). */
  weight?: string;
  /** Continue from an earlier bake (its training state), or a `tinker://…/weights/…` path. */
  start?: BakedModel | string;
  /** How answers are judged at each check (the first decides which checkpoint is kept). */
  criteria?: readonly m.Criterion[];
  /** A metric of (row, outputs), when there is no criterion; else exact match on the rows' answers. */
  metric?: (row: Rec, outputs: Rec) => number | Promise<number>;
  /** `rl`: what an answer is worth. */
  reward?: Reward;
  /** `prefer`: the pairs. */
  pairs?: readonly PreferencePair[];
  /** `sft`: answers the rows that have none (a model name, or an AI function). `distill`: the model whose tokens' probabilities the student learns (a Tinker base model, or a baked model, sharing the student's tokenizer). */
  teacher?: string | BakedModel | Fn;
  /** Rows to check on (default: a share of the rows, held out by design 20's rule). */
  test?: readonly Rec[];
  /** A column whose values stay on one side together (two rows of one screen are both held out, or both trained on). */
  heldOutBy?: string;
  /** The share of cases held out when `test` is not given (default 0.1). */
  heldOut?: number;
  /** At most this many held-out cases at each check (default 100). */
  checkCases?: number;
  /** Temperatures tried at each check (default [0, 0.7, 1]). */
  temperatures?: readonly number[];
  /** Steps between checks (default: about 10 checks a run). */
  evalEvery?: number;
  epochs?: number;
  lr?: number;
  /** LoRA rank (default 32). */
  rank?: number;
  /** Examples (sft, decision), cases (rl, distill) or pairs (prefer) per step. */
  batch?: number;
  /** `rl`, `distill`: answers per case in a step (default 4). */
  group?: number;
  /** `rl`: Tinker's loss (default importance_sampling). */
  loss?: "importance_sampling" | "ppo" | "cispo" | "dro" | "gspo" | "stable_reinforce";
  /** `rl`: divide advantages by the group's standard deviation (default true). */
  normalize?: boolean;
  /** `rl`: KL to the starting weights, folded into the advantages (default 0). */
  kl?: number;
  /** `rl`, `distill`: the sampling temperature of training answers (default 1). */
  temperature?: number;
  /** `prefer`: DPO's β (default 0.1). */
  beta?: number;
  /** Inkling's thinking effort, trained and called (default 0: no hidden thinking). */
  effort?: number | string;
  /** The longest reply sampled (default: 1.25× the longest answer, at least 64; 1024 when none is known). */
  maxTokens?: number;
  /** Keep the training state of the weights the model answers with (the best check's), so `start` continues from exactly them (default true). */
  keepState?: boolean;
  /** The run's folder (default ~/.cache/functai/bakes/<name>-<student>-tinker-<fingerprint>). */
  folder?: string;
  /** Where the model is written (default <folder>/baked). */
  path?: string;
  name?: string;
  seed?: number;
  /** Decide and price everything; spend nothing. */
  planOnly?: boolean;
  /** Tinker's client, or its options (default: TINKER_API_KEY). */
  tinker?: Tinker | TinkerOptions;
  /** Where a student's files come from (a folder; tests): for every model, or by model id. */
  studentFiles?: StudentOptions | ((id: string) => StudentOptions);
  /** A line of progress (default: stderr; false: silent). */
  log?: ((line: string) => void) | false;
  /**
   * Stop at the end of the step under way: the training state is saved (kept a day, as intermediate states are), and
   * the bake ends with `BakeError` code `bake-stopped`. Baking again with the same options and folder goes on from
   * that step. A host that must stop a long run (a server restarting) loses no step.
   */
  signal?: AbortSignal;
}

// ------------------------------------------------------------------ prices

/** Dollars per million tokens. */
export interface Price {
  prefill: number;
  cached: number;
  sample: number;
  train: number;
  context: number | null;
}

const TINKER_MODELS = "https://tinker-docs.thinkingmachines.ai/tinker/models.json";

/** Tinker's price table (its models.json, kept a day), by Tinker id; null when it cannot be read. */
export async function tinkerPrices(): Promise<Record<string, Price> | null> {
  const fs = builtin("node:fs")!;
  const path = builtin("node:path")!;
  const os = builtin("node:os")!;
  const e = env();
  const file = path.join(e["XDG_CACHE_HOME"] ?? path.join(os.homedir(), ".cache"), "functai", "prices", "tinker-models.json");
  let data: unknown = null;
  const fresh = fs.existsSync(file) && Date.now() - fs.statSync(file).mtimeMs < 86_400_000;
  if (fs.existsSync(file) && (fresh || e["FUNCTAI_PRICES"] === "off")) data = parseForeign(fs.readFileSync(file, "utf8"));
  else if (e["FUNCTAI_PRICES"] !== "off") {
    try {
      const r = await fetch(TINKER_MODELS, { signal: AbortSignal.timeout(5000) });
      if (r.ok) {
        const text = await r.text();
        data = parseForeign(text);
        fs.mkdirSync(path.dirname(file), { recursive: true });
        fs.writeFileSync(file, text);
      }
    } catch {
      data = fs.existsSync(file) ? parseForeign(fs.readFileSync(file, "utf8")) : null;
    }
  }
  if (!Array.isArray(data)) return null;
  const dollars = (v: unknown) => (typeof v === "string" ? Number(v.replace(/[$,]/g, "")) : typeof v === "number" ? v : NaN);
  const out: Record<string, Price> = {};
  for (const x of data as Rec[]) {
    const ctx = String(x["context"] ?? "");
    out[x["tinker_id"] as string] = {
      prefill: dollars(x["prefill"]), cached: dollars(x["cached_prefill"] ?? x["prefill"]), sample: dollars(x["sample"]), train: dollars(x["train"]),
      context: ctx ? Math.round(Number(ctx.replace(/[Kk]$/, "")) * (/[Kk]$/.test(ctx) ? 1024 : 1)) : null,
    };
  }
  return out;
}

// ------------------------------------------------------------------ cases

/** A row, read: its inputs (bound as a call binds them), its answer when known, its case key and side. */
interface Case {
  readonly row: Rec;
  readonly inputs: Rec;
  readonly outputs: Rec | null;
  readonly key: string;
  readonly side: "train" | "held_out";
  readonly weight: number;
}

function sideOf(row: Rec, fn: Fn, opts: TrainOptions): "train" | "held_out" {
  const share = opts.heldOut ?? 0.1;
  if (opts.heldOutBy) {
    if (!Object.hasOwn(row, opts.heldOutBy)) throw new BakeError("bake-rows", `heldOutBy: a row has no ${opts.heldOutBy}`);
    // the column's value, as canonical JSON (calls.md), groups the rows (training.md §10)
    return m.splitOf({ conversation: `${opts.heldOutBy}:${lmcc.canonicalJson(toJson(row[opts.heldOutBy])[0])}` }, [], { share });
  }
  return m.splitOf(row, fn, { share });
}

function readCases(fn: Fn, entry: BakeEntry, rows: readonly Rec[], opts: TrainOptions, side?: "held_out"): Case[] {
  const names = fn.interface.inputs.map((f) => f.name);
  return rows.map((row, i) => {
    const given: Rec = {};
    for (const n of names) if (Object.hasOwn(row, n) && row[n] !== undefined) given[n] = row[n];
    for (const [k, v] of Object.entries(opts.fixed ?? {})) if (!Object.hasOwn(given, k)) given[k] = v;
    let inputs: Rec;
    try {
      inputs = checkInputs(fn.interface, given, fn.name);
    } catch (err) {
      throw new BakeError("bake-rows", `row ${i + 1}: ${(err as Error).message}`);
    }
    const outs = entry.outputs.filter((k) => Object.hasOwn(row, k) && row[k] !== null && row[k] !== undefined);
    const outputs = outs.length ? Object.fromEntries(outs.map((k) => [k, row[k]])) : null;
    return { row, inputs, outputs, key: m.caseKey(fn, row), side: side ?? sideOf(row, fn, opts), weight: opts.weight ? Number(row[opts.weight]) : 1 };
  });
}

/** One case as the student reads it: its conversation, pictures, and reply (when outputs are given). */
function conversation(entry: BakeEntry, inputs: Rec, outputs: Rec | null): { messages: Rec[]; pictures: Picture[]; reply: string | null } {
  const plan = studentPlan(entry);
  const left = [...Object.keys(entry.fixed), ...Object.keys(entry.derived)];
  const values = sig.prepareInputs(entry.signature, Object.fromEntries(Object.entries(inputs).filter(([k]) => !left.includes(k))));
  const { messages, pictures } = studentConversation(plan.render(plan.turn(values)).request("student") as Rec);
  if (!outputs) return { messages, pictures, reply: null };
  const shown = Object.fromEntries(entry.outputs.filter((k) => Object.hasOwn(outputs, k)).map((k) => [k, toJson(outputs[k])[0]]));
  const both = studentConversation(plan.render(plan.turn(values), { turns: [plan.example(values, shown)] }).request("student") as Rec);
  const reply = both.messages.filter((x) => x["role"] === "assistant").at(-1)!["content"] as string;
  return { messages, pictures, reply };
}

/** A reply's outputs as the function's layout reads it, or null when it cannot be read. */
function readOutputs(entry: BakeEntry, text: string, cut: boolean): Rec | null {
  if (cut) return null;
  try {
    const response = Response.fromJSON({ model: "student", message: { role: "assistant", parts: [{ type: "text", text }] }, finish_reason: "stop", usage: {} } as never);
    return bridge.read(studentPlan(entry), response).values as Rec;
  } catch {
    return null;
  }
}

// ------------------------------------------------------------------ the student on Tinker, called

/** Calls of a baked student on Tinker: the request rendered with the student's files, sampled, its reply read (training.md §5). */
export class TinkerRunner implements Router {
  readonly student: Student;
  readonly tinker: Tinker;
  readonly temperature: number;
  readonly maxTokens: number;
  /** A decision model: the choices and each one's reply; a call answers the likeliest. */
  readonly decision: { readonly choices: readonly unknown[]; readonly replies: readonly string[] } | null;
  /** When set, each request samples with a seed made from it and the request (the same request, the same answer): checks are reproducible. */
  readonly seed: number | null;
  /** Every call that failed here (Tinker refused or could not be reached; never a reply the function cannot read):
   * a check stops on one instead of scoring the case 0. */
  readonly failures: unknown[] = [];
  readonly #make: () => Promise<Sampler>;
  #ready: Promise<Sampler> | null = null;

  /**
   * `sampler`: the weights' sampler, or a function opening it. A function is called at the first call (a model that is
   * loaded and never called reaches nothing: a program built, read, or measured on other models), and again after
   * a failure to open it (the next call tries again).
   */
  constructor(f: { student: Student; tinker: Tinker; sampler: Promise<Sampler> | Sampler | (() => Promise<Sampler>); temperature: number; maxTokens: number; decision?: { choices: readonly unknown[]; replies: readonly string[] } | null; seed?: number | null }) {
    this.student = f.student;
    this.tinker = f.tinker;
    this.temperature = f.temperature;
    this.maxTokens = f.maxTokens;
    this.decision = f.decision ?? null;
    this.seed = f.seed ?? null;
    const given = f.sampler;
    this.#make = typeof given === "function" ? given : () => Promise.resolve(given);
  }

  /** The sampler, opened at the first call. */
  #sampler(): Promise<Sampler> {
    return (this.#ready ??= this.#make().catch((err: unknown) => { this.#ready = null; throw err; }));
  }

  /** A decision model's probability for each choice, for a request (one sampling request, nothing generated). */
  async probabilities(request: Request): Promise<{ choice: unknown; probability: number }[]> {
    const d = this.decision;
    if (!d) throw new BakeError("baked-format", "this model answers by writing a reply; probabilities are a decision model's");
    const { Request: Req } = await import("@lm15/lm15");
    const { messages, pictures } = studentConversation(Req.toJSON(request) as Rec);
    const point = choicePoint(this.student, messages, pictures, d.replies);
    const rows = Array.from({ length: chunksLength(point.prefix) }, (_, i) => (i === chunksLength(point.prefix) - 1 ? point.tokens : point.tokens.map(() => -1)));
    const r = await (await this.#sampler()).sample(wireChunks([...point.prefix, { text: [point.tokens[0]!] }], pictures), { maxTokens: 1, targetPromptLogprobs: rows });
    const p = T.softmax(r.targetPromptLogprobs!.at(-1)!);
    return d.choices.map((choice, i) => ({ choice, probability: p[i]! }));
  }

  resolve(model: string) {
    return { provider: "tinker", model };
  }

  async complete(request: Request): Promise<Response> {
    try {
      return await this.#answer(request);
    } catch (err) {
      this.failures.push(err);
      throw err;
    }
  }

  async #answer(request: Request): Promise<Response> {
    if (this.decision) {
      const probs = await this.probabilities(request);
      const best = probs.reduce((a, b, i) => (b.probability > probs[a]!.probability ? i : a), 0);
      return Response.fromJSON({ model: "student", message: { role: "assistant", parts: [{ type: "text", text: this.decision.replies[best]! }] }, finish_reason: "stop", usage: {} } as never);
    }
    const { Request: Req } = await import("@lm15/lm15");
    const { messages, pictures } = studentConversation(Req.toJSON(request) as Rec);
    const { prompt } = this.student.render(messages, pictures);
    const sampler = await this.#sampler();
    const limit = request.config?.maxTokens ?? this.maxTokens;
    const temperature = request.config?.temperature ?? this.temperature;
    const seed = this.seed === null ? undefined : parseInt(lmcc.sha256Hex(`${this.seed}:${temperature}:${writeData(Req.toJSON(request))}`).slice(0, 8), 16);
    const r = await sampler.sample(wireChunks(prompt, pictures), {
      numSamples: 1, maxTokens: limit, temperature, stop: [this.student.stopToken], ...(seed !== undefined ? { seed } : {}),
    });
    const seq = r.sequences[0]!;
    return Response.fromJSON({
      model: "student", message: { role: "assistant", parts: [{ type: "text", text: this.student.readReply(seq.tokens) }] },
      finish_reason: seq.stopReason === "stop" ? "stop" : "length", usage: { input_tokens: chunksLength(prompt), output_tokens: seq.tokens.length },
    } as never);
  }
}

/**
 * What a run spent: the plan's estimate, and what Tinker billed this run's sessions (other work on the account
 * in the same hours is not counted). Tinker's records come hours later: `billed` is null while there are none,
 * and `billed_through` the end of the last hour billed so far (later hours may still come).
 */
export function spentOf(estimated: unknown, b: { events: readonly Rec[] }, sessions: readonly string[]): Rec {
  const mine = new Set(sessions);
  const events = b.events.filter((e) => mine.has(String(e["session_id"] ?? "")));
  if (!events.length) return { estimated, billed: null, billed_through: null };
  const billed = events.reduce((sum, e) => sum + Number(e["estimated_cost_usd"] ?? 0), 0);
  const ends = events.map((e) => String(e["bucket_end"] ?? "")).filter((x) => !Number.isNaN(Date.parse(x)));
  const through = ends.length ? ends.reduce((a, b) => (Date.parse(b) > Date.parse(a) ? b : a)) : null;
  return { estimated, billed, billed_through: through && new Date(through).toISOString().replace(/\.\d{3}Z$/, "Z") };
}

/** The files option for one model. */
function filesFor(f: TrainOptions["studentFiles"], id: string): StudentOptions {
  return typeof f === "function" ? f(id) : f ?? {};
}

function tinkerOf(t: TrainOptions["tinker"], meta?: Record<string, string>): Tinker {
  return t instanceof Tinker ? t : new Tinker({ ...(t ?? {}), userMetadata: { ...(meta ?? {}), ...((t as TinkerOptions | undefined)?.userMetadata ?? {}) } });
}

/**
 * A model trained on Tinker, from its folder: calls render with the student's
 * published files (refused when they changed since training: the tokens would
 * differ) and sample its kept weights.
 */
export async function bakedOnTinker(folder: string, opts: { tinker?: Tinker | TinkerOptions; studentFiles?: TrainOptions["studentFiles"]; student?: Student } = {}): Promise<BakedModel> {
  const fs = builtin("node:fs")!;
  const path = builtin("node:path")!;
  const file = path.join(folder, "baked.json");
  if (!fs.existsSync(file)) throw new BakeError("baked-format", `${folder} has no baked.json`);
  const meta = parseData(fs.readFileSync(file, "utf8")) as Rec;
  if (meta["functai_baked"] !== BAKED_FORMAT) throw new BakeError("baked-format", `${folder} is not baked.json format ${BAKED_FORMAT}`);
  const w = meta["weights"] as Rec;
  if (w["form"] !== "remote" || w["service"] !== "tinker") throw new BakeError("baked-format", `${folder}'s weights are not on Tinker; baked(folder, { url }) runs weights a server serves`);
  const tpl = meta["template"] as Rec;
  const student = opts.student && opts.student.id === meta["student"] ? opts.student
    : await Student.load(meta["student"] as string, { ...filesFor(opts.studentFiles, meta["student"] as string), effort: Number((tpl["kwargs"] as Rec | undefined)?.["reasoning_effort"] ?? 0) });
  if (student.templateSha256 !== tpl["sha256"] || (tpl["tokenizer_sha256"] && student.tokenizerSha256 !== tpl["tokenizer_sha256"])) {
    throw new BakeError("baked-format", `${meta["student"]}'s published files changed since this model was trained (its template or tokenizer): calls would not send the tokens it learned`);
  }
  const tinker = tinkerOf(opts.tinker, { functai: "call", baked: String(meta["name"]) });
  const decision = ((meta["training"] as Rec | undefined)?.["decision"] ?? null) as { choices: unknown[]; replies: string[] } | null;
  // its sampler opened at its first call: loading a model reaches nothing
  const runner = new TinkerRunner({ student, tinker, sampler: () => tinker.sampler({ path: w["uri"] as string }), temperature: Number((meta["sampling"] as Rec | undefined)?.["temperature"] ?? 0),
    maxTokens: Number((meta["generation"] as Rec | undefined)?.["max_new_tokens"] ?? 1024), decision });
  return new BakedModel({ folder: path.resolve(folder), meta, entries: new Map(((meta["functions"] ?? []) as Rec[]).map((d) => [d["name"] as string, entryOf(d)])),
    url: "", model: String(meta["name"]), apiKey: null, timeout: 600_000, runner });
}

// ------------------------------------------------------------------ the plan

/** Everything a bake will do, decided and priced before anything is spent. */
export class TrainPlan {
  readonly fn: Fn;
  readonly opts: TrainOptions;
  readonly method: Method;
  readonly entry: BakeEntry;
  readonly student: Student;
  readonly train: Case[];
  readonly held: Case[];
  readonly settings: Rec;
  readonly tokens: Rec;
  readonly dollars: Rec;
  readonly notes: string[];
  readonly fingerprint: string;
  readonly folder: string;
  readonly output: string;
  readonly name: string;
  /** @internal */ readonly rendered: Map<string, { chunks: Chunk[]; replyStart: number; pictures: Picture[]; prompt: Chunk[] }>;

  constructor(f: Omit<TrainPlan, "toString" | "toJSON" | "run">) {
    this.fn = f.fn;
    this.opts = f.opts;
    this.method = f.method;
    this.entry = f.entry;
    this.student = f.student;
    this.train = f.train;
    this.held = f.held;
    this.settings = f.settings;
    this.tokens = f.tokens;
    this.dollars = f.dollars;
    this.notes = f.notes;
    this.fingerprint = f.fingerprint;
    this.folder = f.folder;
    this.output = f.output;
    this.name = f.name;
    this.rendered = f.rendered;
  }

  toString(): string {
    const s = this.settings;
    const money = (x: unknown) => (typeof x === "number" ? `$${x < 1 ? x.toFixed(2) : x.toFixed(2)}` : "price unknown");
    const lines = [
      `bake ${this.name} → ${this.student.id} on Tinker (${this.method}, LoRA r${s["rank"]}${this.student.kwargs["reasoning_effort"] !== undefined ? `, effort ${this.student.kwargs["reasoning_effort"]}` : ""})`,
      `  cases      ${this.train.length.toLocaleString()} train · ${this.held.length.toLocaleString()} held out${this.opts.heldOutBy ? ` by ${JSON.stringify(this.opts.heldOutBy)}` : ""}`,
      `  tokens     ${Number(this.tokens["train_per_pass"] ?? 0).toLocaleString()} trained per pass · replies up to ${s["max_tokens"]} tokens`,
      `  training   ${s["epochs"]} pass(es), ${s["steps"]} steps of ${s["batch"]}, lr ${Number(s["lr"]).toExponential(1)}${this.method === "rl" ? `, ${s["group"]} answers per case, ${s["loss"]}` : ""}`,
      `  checks     every ${s["eval_every"]} steps, ${s["check_cases"]} cases at temperatures ${(s["temperatures"] as number[]).join(", ")}`,
      `  money      ≈ ${money(this.dollars["total"])} (training ${money(this.dollars["training"])} · sampling ${money(this.dollars["sampling"])} · checks ${money(this.dollars["checks"])})`,
      `  your data  ${this.train.length + this.held.length} cases${this.rendered.size && [...this.rendered.values()].some((r) => r.pictures.length) ? " and their pictures" : ""} are sent to Thinking Machines`,
      `  run        ${this.folder}`,
    ];
    for (const n of this.notes) lines.push(`  note: ${n}`);
    return lines.join("\n");
  }

  toJSON(): Rec {
    return { name: this.name, student: this.student.id, method: this.method, where: "tinker", fingerprint: this.fingerprint, settings: this.settings, tokens: this.tokens,
      estimate: this.dollars, functions: [entryJson(this.entry)], template: this.student.describe(), output: this.output, notes: this.notes };
  }

  /** Do it: train, check, keep the best, write the model. */
  run(): Promise<BakedModel> {
    return new TrainRun(this).go();
  }
}

/** The settings a method starts from (each can be set). */
function defaults(method: Method, n: number, tokens: number, o: TrainOptions): Rec {
  const lr = o.lr ?? { sft: 1e-4, rl: 4e-5, prefer: 1e-5, distill: 1e-4, decision: 1e-4 }[method];
  const batch = o.batch ?? (method === "sft" || method === "decision" ? Math.max(1, Math.min(n, Math.max(8, Math.min(32, Math.round(n / 100))))) : method === "prefer" ? Math.min(16, n) : Math.min(32, n));
  const epochs = o.epochs ?? (method === "sft" ? (tokens > 20e6 ? 1 : n < 1000 ? 3 : 2) : method === "rl" ? 2 : 1);
  const perEpoch = Math.max(1, Math.floor(n / batch));
  const steps = perEpoch * epochs;
  return { lr, batch, epochs, steps, steps_per_epoch: perEpoch, rank: o.rank ?? 32, group: o.group ?? 4, loss: o.loss ?? "importance_sampling",
    normalize: o.normalize ?? true, kl: o.kl ?? 0, temperature: o.temperature ?? 1, beta: o.beta ?? 0.1, seed: o.seed ?? 0,
    eval_every: o.evalEvery ?? Math.max(1, Math.round(steps / 10)), temperatures: [...(o.temperatures ?? [0, 0.7, 1])],
    schedule: method === "sft" ? { warmup: 0.03, decay: 0.2 } : null, keep_state: o.keepState ?? true };
}

function say(o: TrainOptions): (line: string) => void {
  if (o.log === false) return () => {};
  return o.log ?? ((line: string) => process.stderr.write(`[bake] ${line}\n`));
}

/** Decide and price a bake (spends nothing). */
export async function planBake(fn: Fn, rows: readonly Rec[], opts: TrainOptions = {}): Promise<TrainPlan> {
  if ((opts.where ?? "tinker") !== "tinker") throw new BakeError("bake-rows", "TypeScript trains on Tinker (where: \"tinker\"); Python's functai.bake trains here too");
  const method: Method = opts.method ?? "sft";
  const startMeta = opts.start instanceof BakedModel ? opts.start.meta : null;
  const studentId = opts.student ?? (startMeta?.["student"] as string | undefined);
  if (!studentId) throw new BakeError("bake-rows", "bake: name the student (student: \"Qwen/Qwen3.5-4B\"), or start from an earlier bake");
  const entry = bakeEntry(fn as never, { ...opts, rows });
  if (startMeta) {
    const was = ((startMeta["functions"] ?? []) as Rec[]).find((d) => d["name"] === fn.name);
    if (!was || was["fingerprint"] !== entry.fingerprint) throw new BakeError("baked-changed", `start: that model was not trained on ${fn.name} as it is now`);
  }
  const startStudent = opts.start instanceof BakedModel && opts.start.runner instanceof TinkerRunner ? opts.start.runner.student : null;
  const student = startStudent && startStudent.id === studentId && opts.effort === undefined ? startStudent
    : await Student.load(studentId, { ...filesFor(opts.studentFiles, studentId), effort: effortOf(opts.effort ?? (startStudent?.kwargs["reasoning_effort"] as number | undefined)) });
  const notes: string[] = [];
  let cases: Case[];
  let held: Case[];
  if (opts.test) {
    cases = readCases(fn, entry, rows, opts).map((c) => ({ ...c, side: "train" as const }));
    held = readCases(fn, entry, opts.test, opts, "held_out");
  } else {
    const all = readCases(fn, entry, rows, opts);
    cases = all.filter((c) => c.side === "train");
    held = all.filter((c) => c.side === "held_out");
  }
  if (method === "sft" && !opts.teacher) {
    const missing = cases.filter((c) => !c.outputs).length;
    if (missing) notes.push(`${missing} rows without an answer are left out (give teacher: to answer them)`);
    cases = cases.filter((c) => c.outputs);
  }
  if (method === "decision") decisionChoices(fn, entry);            // refuses early when the answers are not a few choices
  if (!cases.length && method !== "prefer") throw new BakeError("bake-rows", "no rows to train on");
  // the conversations' tokens: replies where answers are known
  const rendered = new Map<string, { chunks: Chunk[]; replyStart: number; pictures: Picture[]; prompt: Chunk[] }>();
  let longest = 0;
  let replyMax = 0;
  let trainTokens = 0;
  let promptTokens = 0;
  for (const c of [...cases, ...held]) {
    const conv = conversation(entry, c.inputs, c.outputs);
    const r = student.render(conv.messages, conv.pictures, conv.reply);
    const full = r.full ?? r.prompt;
    rendered.set(c.key + (c.outputs ? `:${valueHash(toJson(c.outputs)[0])}` : ""), { chunks: full, replyStart: r.replyStart, pictures: conv.pictures, prompt: r.prompt });
    longest = Math.max(longest, chunksLength(full));
    if (r.full) replyMax = Math.max(replyMax, chunksLength(r.full) - r.replyStart);
    if (c.side === "train") {
      trainTokens += chunksLength(full) * c.weight;
      promptTokens += chunksLength(r.prompt);
    }
  }
  // per step: examples (sft, decision), pairs (prefer), or distinct cases (rl, distill: rows with the same inputs are one case, their answers its references)
  const n = method === "prefer" ? (opts.pairs ?? []).length : method === "rl" || method === "distill" ? uniqueCases(cases).length : cases.length;
  const s = defaults(method, n, trainTokens, opts);
  s["max_tokens"] = opts.maxTokens ?? (replyMax ? Math.max(64, Math.ceil(replyMax * 1.25)) : 1024);
  s["check_cases"] = Math.min(held.length, opts.checkCases ?? 100);
  s["effort"] = effortOf(opts.effort);
  if (!held.length) notes.push("no held-out cases: nothing to choose a checkpoint by; the last one is kept, at temperature 0");
  const prices = await tinkerPrices();
  const price = prices?.[studentId] ?? null;
  if (prices && !price) throw new BakeError("bake-rows", `Tinker does not train ${studentId} (it trains ${Object.keys(prices).filter((k) => !k.includes(":")).slice(0, 8).join(", ")}…)`);
  if (price?.context && longest > price.context) throw new BakeError("bake-rows", `the longest conversation (${longest} tokens) is longer than Tinker's context for ${studentId} (${price.context})`);
  const answerTokens = replyMax ? replyMax / 2 : Number(s["max_tokens"]) / 2;
  const meanPrompt = cases.length ? promptTokens / cases.length : 0;
  const checks = Math.ceil(Number(s["steps"]) / Number(s["eval_every"])) + 1;
  const checkTokens = Number(s["check_cases"]) * (s["temperatures"] as number[]).length;
  let training = 0, sampling = 0;
  const steps = Number(s["steps"]), batch = Number(s["batch"]), group = Number(s["group"]);
  if (method === "sft") training = trainTokens * Number(s["epochs"]);
  else if (method === "decision") training = trainTokens * Number(s["epochs"]) * 2;
  else if (method === "prefer") training = (opts.pairs ?? []).length * 2 * (meanPrompt + answerTokens) * (Number(s["epochs"]) * 2 + 1);
  else {
    training = steps * batch * group * (meanPrompt + answerTokens);
    sampling = steps * batch * (meanPrompt * (price?.prefill ?? 0) + group * answerTokens * (price?.sample ?? 0)) / 1e6;
    if (method === "distill" || Number(s["kl"]) > 0) sampling += steps * batch * group * (meanPrompt + answerTokens) * (price?.prefill ?? 0) / 1e6;
  }
  const dollars: Rec = price ? {
    training: training * price.train / 1e6, sampling,
    checks: checks * checkTokens * (meanPrompt * price.cached + answerTokens * price.sample) / 1e6,
  } : { training: null, sampling: null, checks: null };
  if (price) dollars["total"] = Number(dollars["training"]) + Number(dollars["sampling"]) + Number(dollars["checks"]);
  else notes.push("Tinker's price table could not be read: no price");
  if (!replyMax && method !== "sft") notes.push("reply lengths are guessed (no known answers); the first check measures them");
  const h = lmcc.sha256({ student: studentId, method, settings: s, entry: entryJson(entry), start: startMeta ? String((startMeta["weights"] as Rec)["state_uri"] ?? "") : typeof opts.start === "string" ? opts.start : null,
    cases: [...cases, ...held].map((c) => [c.key, c.side, c.outputs ? valueHash(toJson(c.outputs)[0]) : null]), pairs: (opts.pairs ?? []).length,
    reward: opts.reward ? (isGroup(opts.reward) ? opts.reward.name : typeof opts.reward === "function" ? opts.reward.name : (opts.reward as m.Criterion).name) : null } as never).slice(7, 19);
  const path = builtin("node:path")!;
  const os = builtin("node:os")!;
  const e = env();
  const name = opts.name ?? fn.name;
  const folder = path.resolve(opts.folder ?? path.join(e["XDG_CACHE_HOME"] ?? path.join(os.homedir(), ".cache"), "functai", "bakes", `${name}-${studentId.split(":")[0]!.split("/").at(-1)!.toLowerCase()}-tinker-${h}`));
  return new TrainPlan({ fn, opts, method, entry, student, train: cases, held, settings: s, tokens: { train_per_pass: trainTokens, longest, reply_max: replyMax },
    dollars, notes, fingerprint: h, folder, output: path.resolve(opts.path ?? path.join(folder, "baked")), name, rendered });
}

/**
 * Train a student of `fn` on Tinker (see the module's docs); resolves to the
 * model, which `fn.using({ lm: model })` calls. `planOnly`: the plan, nothing spent.
 */
export async function bake(fn: Fn, rows: readonly Rec[], opts: TrainOptions & { planOnly: true }): Promise<TrainPlan>;
export async function bake(fn: Fn, rows: readonly Rec[], opts?: TrainOptions): Promise<BakedModel>;
export async function bake(fn: Fn, rows: readonly Rec[], opts: TrainOptions = {}): Promise<BakedModel | TrainPlan> {
  const plan = await planBake(fn, rows, opts);
  say(opts)(String(plan));
  if (opts.planOnly) return plan;
  return plan.run();
}

// ------------------------------------------------------------------ decisions

/** The choices of a function whose answer is a few allowed values (enum, boolean). */
function decisionChoices(fn: Fn, entry: BakeEntry): unknown[] {
  const out = fn.interface.outputs.at(-1)!;
  const shape = out.shape as Rec;
  if (Array.isArray(shape["enum"])) return [...shape["enum"] as unknown[]];
  if (shape["type"] === "boolean") return [true, false];
  throw new BakeError("bake-rows", `method "decision" is for a function whose answer is one of a few values (an enum, a boolean); ${fn.name}'s ${out.name} is not (entry ${entry.name})`);
}

/** The reply each choice is written as (the layout's reply for that answer). */
function choiceReplies(entry: BakeEntry, answerName: string, choices: readonly unknown[], inputs: Rec): string[] {
  return choices.map((ch) => conversation(entry, inputs, { [answerName]: ch }).reply!);
}

/** Where the choices' replies first differ after a prompt, and each choice's token there (refused when one token does not tell them apart). */
function choicePoint(student: Student, messages: readonly Rec[], pictures: readonly Picture[], replies: readonly string[]): { prefix: Chunk[]; tokens: number[] } {
  const rendered = replies.map((reply) => student.render(messages, pictures, reply));
  const flats = rendered.map((r) => T.positions(r.full!));
  let at = rendered[0]!.replyStart;
  while (flats.every((f) => f[at] !== undefined && f[at] === flats[0]![at])) at++;
  const tokens = flats.map((f) => f[at] as number);
  if (tokens.some((t) => t === null || t === undefined) || new Set(tokens).size !== tokens.length) {
    throw new BakeError("bake-rows", `the answers' replies are not told apart by one token where they first differ; a decision model needs that`);
  }
  const full = rendered[0]!.full!;
  const prefix: Chunk[] = [];
  let pos = 0;
  for (const c of full) {
    const len = "text" in c ? c.text.length : c.tokens;
    if (pos + len <= at) prefix.push(c);
    else if ("text" in c && pos < at) prefix.push({ text: c.text.slice(0, at - pos) });
    pos += len;
    if (pos >= at) break;
  }
  return { prefix, tokens };
}

/** A training case's decision point: its prompt and the choices' replies. */
function decisionPoint(student: Student, entry: BakeEntry, answerName: string, choices: readonly unknown[], inputs: Rec): { prefix: Chunk[]; tokens: number[]; pictures: Picture[] } {
  const conv = conversation(entry, inputs, null);
  const point = choicePoint(student, conv.messages, conv.pictures, choiceReplies(entry, answerName, choices, inputs));
  return { ...point, pictures: conv.pictures };
}

// ------------------------------------------------------------------ the run

interface RunState {
  state: "running" | "done" | "failed" | "stopped";
  step: number;
  model_id?: string;
  states: { step: number; path: string }[];
  best: { step: number; temperature: number; score: number; sampler: string | null; state?: string | null } | null;
  checks: { step: number; scores: Record<string, number | null>; agreement?: number | null }[];
  spent: number;
  started: string;
  /** The Tinker sessions that worked on it (one per process; a resumed run has several): its bill is theirs. */
  sessions?: string[];
  error?: string;
}

class TrainRun {
  readonly plan: TrainPlan;
  readonly say: (line: string) => void;
  readonly fs = builtin("node:fs")!;
  readonly path = builtin("node:path")!;
  tinker!: Tinker;
  run!: TrainingRun;
  st!: RunState;
  started = Date.now();

  constructor(plan: TrainPlan) {
    this.plan = plan;
    this.say = say(plan.opts);
  }

  // ---- the folder

  file(name: string): string {
    return this.path.join(this.plan.folder, name);
  }

  save(): void {
    this.fs.writeFileSync(this.file("run.json.tmp"), writeData(this.st, 1) + "\n");
    this.fs.renameSync(this.file("run.json.tmp"), this.file("run.json"));
  }

  metric(rec: Rec): void {
    this.fs.appendFileSync(this.file("metrics.jsonl"), writeData({ ...rec, seconds: Math.round((Date.now() - this.started) / 100) / 10 }) + "\n");
  }

  // ---- doing

  async go(): Promise<BakedModel> {
    const p = this.plan;
    this.fs.mkdirSync(p.folder, { recursive: true });
    const done = this.path.join(p.output, "baked.json");
    if (this.fs.existsSync(this.file("run.json"))) {
      this.st = parseData(this.fs.readFileSync(this.file("run.json"), "utf8")) as RunState;
      if (this.st.state === "done" && this.fs.existsSync(done)) {
        this.say(`already baked: ${p.output}`);
        return bakedOnTinker(p.output, { tinker: p.opts.tinker, studentFiles: p.opts.studentFiles, student: p.student });
      }
    } else {
      this.st = { state: "running", step: 0, states: [], best: null, checks: [], spent: 0, started: new Date().toISOString() };
      this.fs.writeFileSync(this.file("plan.json"), writeData(p.toJSON(), 1) + "\n");
    }
    this.tinker = tinkerOf(p.opts.tinker, { functai: "bake", bake: p.name, fingerprint: p.fingerprint });
    try {
      this.st.sessions = [...new Set([...(this.st.sessions ?? []), await this.tinker.session()])];
      this.save();
      await this.open();
      if (p.method === "sft") await this.sft();
      else if (p.method === "rl" || p.method === "distill") await this.rollouts();
      else if (p.method === "prefer") await this.prefer();
      else await this.decision();
      const model = await this.finish();
      this.st.state = "done";
      this.save();
      return model;
    } catch (err) {
      this.st.state = (err as { code?: unknown }).code === "bake-stopped" ? "stopped" : "failed";
      this.st.error = `${(err as Error).name}: ${(err as Error).message}`;
      this.save();
      throw err;
    } finally {
      this.tinker.close();
    }
  }

  /** The training run: continued from the run's last saved state, the start's state, or new. */
  async open(): Promise<void> {
    const p = this.plan;
    const last = this.st.states.at(-1);
    const start = p.opts.start;
    if (last) {
      this.say(`resuming from step ${last.step}`);
      this.run = await this.tinker.loadState(last.path, { optimizer: true, base: p.student.id });
      this.st.step = last.step;
    } else if (start) {
      const path = typeof start === "string" ? start : String((start.meta["weights"] as Rec)["state_uri"] ?? "");
      if (!path) throw new BakeError("bake-rows", "start: that model kept no training state (bake it with keepState) to continue from");
      this.run = await this.tinker.loadState(path, { optimizer: false, base: p.student.id });
    } else {
      this.run = await this.tinker.createModel(p.student.id, { rank: Number(p.settings["rank"]), seed: Number(p.settings["seed"]), userMetadata: { functai: "bake", bake: p.name } });
    }
    this.st.model_id = this.run.modelId;
    this.save();
  }

  lrAt(step: number): number {
    const s = this.plan.settings;
    const lr = Number(s["lr"]);
    const sched = s["schedule"] as Rec | null;
    if (!sched) return lr;
    const total = Number(s["steps"]);
    const w = Math.max(1, Math.round(total * Number(sched["warmup"])));
    const d = Math.max(1, Math.round(total * Number(sched["decay"])));
    if (step < w) return lr * (step + 1) / w;
    if (step < total - d) return lr;
    return lr * Math.max(0, 1 - Math.sqrt((step - (total - d) + 1) / d));
  }

  /** Shuffled indices of an epoch (seeded). */
  order(n: number, epoch: number): number[] {
    let a = (Number(this.plan.settings["seed"]) * 1_000_003 + epoch + 1) >>> 0;
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

  rendered(c: Case): { chunks: Chunk[]; replyStart: number; pictures: Picture[]; prompt: Chunk[] } {
    const r = this.plan.rendered.get(c.key + (c.outputs ? `:${valueHash(toJson(c.outputs)[0])}` : ""));
    if (r) return r;
    const conv = conversation(this.plan.entry, c.inputs, c.outputs);
    const x = this.plan.student.render(conv.messages, conv.pictures, conv.reply);
    return { chunks: x.full ?? x.prompt, replyStart: x.replyStart, pictures: conv.pictures, prompt: x.prompt };
  }

  /** After a step: checks when due, and an intermediate state now and then (kept a day; the last one only). */
  async afterStep(step: number): Promise<void> {
    const s = this.plan.settings;
    this.st.step = step;
    if (step % Number(s["eval_every"]) === 0 || step === Number(s["steps"])) await this.check(step);
    const every = Math.max(1, Math.round(Number(s["steps"]) / 4));
    const stop = this.plan.opts.signal?.aborted === true && step < Number(s["steps"]);
    if ((step % every === 0 || stop) && step < Number(s["steps"]) && this.st.states.at(-1)?.step !== step) {
      const path = await this.run.saveState(`step-${String(step).padStart(6, "0")}`, { ttlSeconds: 86_400 });
      const previous = this.st.states.at(-1);
      this.st.states.push({ step, path });
      if (previous) await this.tinker.deleteCheckpoint(previous.path).catch(() => {});
    }
    this.save();
    if (stop) throw new BakeError("bake-stopped", `stopped after step ${step} of ${s["steps"]}: its state is saved (kept a day); bake again with the same options and folder to go on from there`);
  }

  // ---- checks

  /** The deciding score of the current weights on the held-out cases, at each temperature; the best kept as a sampler checkpoint. */
  async check(step: number): Promise<void> {
    const p = this.plan;
    if (!p.held.length) return;
    if (this.st.checks.some((c) => c.step === step)) return;
    const sampler = await this.run.sampler();
    const cases = p.held.slice(0, Number(p.settings["check_cases"]));
    const scores: Record<string, number | null> = {};
    let agreement: number | null = null;
    const reward = p.opts.reward;
    if (p.method === "rl" && !p.opts.criteria?.length) {
      // rl is checked as it trains: the mean reward over a group of answers per case (one answer per case is a coin
      // flip at temperature 1), a group reward by its own score
      const up = !isGroup(reward) || reward.higherIsBetter;
      const temps = p.settings["temperatures"] as number[];
      for (const t of temps) {
        const got = await this.groups(sampler, cases, t, Number(p.settings["group"]), step);
        scores[String(t)] = got.length ? got.reduce((a, g) => a + (up ? g.score : -g.score), 0) / got.length : null;
        if (t === temps.at(-1)) agreement = this.agreement(got.map((g) => g.answers));
      }
    } else {
      const model = this.modelOn(sampler, 0, step);
      for (const t of p.settings["temperatures"] as number[]) scores[String(t)] = await this.score(model, cases, t);
      if (p.method === "rl") {
        const got = await this.groups(sampler, cases.slice(0, 20), Number(p.settings["temperature"]), Number(p.settings["group"]), step);
        agreement = this.agreement(got.map((g) => g.answers));
      }
    }
    this.st.checks.push({ step, scores, agreement });
    this.metric({ step, check: scores, ...(agreement !== null ? { agreement } : {}) });
    this.say(`check at step ${step}: ${Object.entries(scores).map(([t, v]) => `T${t} ${v === null ? "–" : v.toFixed(3)}`).join(", ")}${agreement !== null ? `; answers alike ${agreement.toFixed(2)}` : ""}`);
    const best = T.choose(this.st.checks);
    if (best && best.step === step && (!this.st.best || this.st.best.step !== step)) {
      // the best weights so far: kept for sampling, and (to continue from them) their training state; the previous best's go
      const name = `best-step-${String(step).padStart(6, "0")}`;
      const path = await this.run.saveForSampler(name);
      const state = p.settings["keep_state"] ? await this.run.saveState(`${name}-state`) : null;
      const old = this.st.best;
      this.st.best = { ...best, sampler: path, state };
      if (old?.sampler) await this.tinker.deleteCheckpoint(old.sampler).catch(() => {});
      if (old?.state) await this.tinker.deleteCheckpoint(old.state).catch(() => {});
    } else if (best && this.st.best) {
      this.st.best = { ...best, sampler: this.st.best.sampler, state: this.st.best.state ?? null };
    }
    this.save();
  }

  /** Share of pairs of answers to one case that are the same (design 20's self-agreement), over cases. */
  agreement(groups: readonly (readonly unknown[])[]): number | null {
    const per: number[] = [];
    for (const g of groups) {
      const xs = g.filter((x) => x !== null).map((x) => writeData(toJson(x)[0]));
      let same = 0, n = 0;
      for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) {
        n++;
        if (xs[i] === xs[j]) same++;
      }
      if (n) per.push(same / n);
    }
    return per.length ? per.reduce((a, b) => a + b, 0) / per.length : null;
  }

  /** The function on given weights: a baked model whose calls sample them. */
  modelOn(sampler: Sampler, temperature: number, step: number): BakedModel {
    const p = this.plan;
    const meta = this.meta(null);
    const decision = ((meta["training"] as Rec)["decision"] ?? null) as { choices: unknown[]; replies: string[] } | null;
    const runner = new TinkerRunner({ student: p.student, tinker: this.tinker, sampler, temperature, maxTokens: Number(p.settings["max_tokens"]), decision, seed: this.seedFor(step, "check") });
    return new BakedModel({ folder: p.folder, meta, entries: new Map([[p.entry.name, p.entry]]), url: "", model: p.name, apiKey: null, timeout: 600_000, runner });
  }

  /** The deciding score of a model on cases at a temperature: the first criterion, else the metric, else exact match. */
  async score(model: BakedModel, cases: readonly Case[], temperature: number): Promise<number | null> {
    const p = this.plan;
    // never the reply cache: a request a check repeats must be answered by these weights, not an earlier check's
    const fn = p.fn.using({ lm: model as never, temperature, cacheReplies: false } as never);
    const rows = cases.map((c) => c.row);
    const crit = p.opts.criteria?.[0] ?? (p.method === "rl" && p.opts.reward && !isGroup(p.opts.reward) && typeof p.opts.reward !== "function" ? p.opts.reward as m.Criterion : null);
    const ev = crit ? await evaluate(fn as never, rows as never, { criteria: [crit], progress: false, concurrency: 16 })
      : p.opts.metric ? await evaluate(fn as never, rows as never, { metric: p.opts.metric as never, progress: false, concurrency: 16 })
      : await evaluate(fn as never, rows as never, { progress: false, concurrency: 16 });
    // the deciding metric by its name: rows that know their answers add exact match too, first
    // a call that failed in Tinker or on the way says nothing of the weights: stop (the run resumes), never score it 0
    const failed = (model.runner as TinkerRunner | null)?.failures ?? [];
    if (failed.length) throw failed[0];
    const name = crit ? crit.name : p.opts.metric ? (p.opts.metric.name || "metric") : ev.metrics[0]!;
    return ev.summary.find((x) => x.metric === name)?.mean ?? null;
  }

  /** Answers to cases, `group` each, read and rewarded (rl's rollouts and checks). */
  /** A sample's seed: the bake's seed, the step, the case (a run samples the same answers again). */
  seedFor(step: number, key: string): number {
    return parseInt(lmcc.sha256Hex(`${this.plan.settings["seed"]}:${step}:${key}`).slice(0, 8), 16);
  }

  async groups(sampler: Sampler, cases: readonly Case[], temperature: number, group: number, step = this.st.step): Promise<{ c: Case; answers: (Rec | null)[]; seqs: { tokens: number[]; logprobs: number[] }[]; rewards: number[]; score: number; prompt: Chunk[]; pictures: Picture[] }[]> {
    const p = this.plan;
    const answerName = p.fn.interface.outputs.at(-1)!.name;
    const byKey = new Map<string, unknown[]>();
    for (const c of [...p.train, ...p.held]) if (c.outputs && Object.hasOwn(c.outputs, answerName)) byKey.set(c.key, [...(byKey.get(c.key) ?? []), toJson(c.outputs[answerName])[0]]);
    return await pool(cases, 16, async (c) => {
      const r = this.rendered(c);
      const res = await sampler.sample(wireChunks(r.prompt, r.pictures), { numSamples: group, temperature, maxTokens: Number(p.settings["max_tokens"]), stop: [p.student.stopToken], seed: this.seedFor(step, c.key) });
      const seqs = res.sequences.map((s) => ({ tokens: s.tokens, logprobs: s.logprobs ?? s.tokens.map(() => 0), cut: s.stopReason !== "stop" }));
      const answers = seqs.map((s) => readOutputs(p.entry, p.student.readReply(s.tokens), s.cut));
      const refs = byKey.get(c.key) ?? [];
      const { rewards, score } = await this.reward(c, answers, refs);
      return { c, answers, seqs, rewards, score, prompt: r.prompt, pictures: r.pictures };
    });
  }

  /** Rewards of a group of answers (training.md §7): a criterion's value, a function's number, or a group reward; unreadable answers the worst. */
  async reward(c: Case, answers: readonly (Rec | null)[], references: readonly unknown[]): Promise<{ rewards: number[]; score: number }> {
    const p = this.plan;
    const reward = p.opts.reward;
    const answerName = p.fn.interface.outputs.at(-1)!.name;
    if (!reward) throw new BakeError("bake-rows", "method \"rl\" needs a reward: a criterion with a judge, a function, or a group reward (rewards.energy)");
    if (isGroup(reward)) return reward.rewards(answers.map((a) => (a ? a[answerName] : null)), references, c.inputs);
    const inputs = Object.fromEntries(Object.entries(c.inputs).map(([k, v]) => [k, toJson(v)[0]]));
    const expected = c.outputs ? toJson(c.outputs[answerName])[0] : undefined;
    const values = await Promise.all(answers.map(async (a, i) => {
      if (!a) return 0;
      const args = { ...inputs, inputs, outputs: a, answer: a[answerName], expected, references: [...references], answers: answers.map((x) => (x ? x[answerName] : null)), index: i };
      if (typeof reward === "function") return Number(await reward(args as never));
      const crit = reward as m.Criterion;
      const [got] = await m.askJudge(crit.judge!, args);
      return m.numberOf({ kind: crit.kind, value: m.judgedValue(m.criterionData(crit), got), answers: crit.answers as never, scale: crit.scale as never }) ?? 0;
    }));
    return { rewards: values, score: values.reduce((a, b) => a + b, 0) / Math.max(1, values.length) };
  }

  // ---- the methods

  async sft(): Promise<void> {
    const p = this.plan;
    const s = p.settings;
    let cases = p.train;
    if (p.opts.teacher && cases.some((c) => !c.outputs)) cases = await this.labelled(cases);
    const steps = Number(s["steps"]), per = Number(s["batch"]), perEpoch = Number(s["steps_per_epoch"]);
    if (this.st.step === 0) await this.check(0);
    for (let step = this.st.step; step < steps; step++) {
      const epoch = Math.floor(step / perEpoch);
      const k = step % perEpoch;
      const idx = this.order(cases.length, epoch).slice(k * per, (k + 1) * per);
      const data = idx.map((i) => {
        const r = this.rendered(cases[i]!);
        return wireDatum(T.sftDatum(r.chunks, r.replyStart, cases[i]!.weight), r.pictures);
      });
      const [out] = await both(this.run.forwardBackward(data, "cross_entropy"), this.run.optimStep({ learning_rate: this.lrAt(step) }));
      let num = 0, den = 0;
      out.outputs.forEach((o, i) => {
        const w = data[i]!.inputs["weights"] as number[];
        (o.logprobs as number[]).forEach((lp, j) => { num -= lp * w[j]!; den += w[j]!; });
      });
      this.metric({ step: step + 1, loss: den ? num / den : null, lr: this.lrAt(step), examples: data.length });
      await this.afterStep(step + 1);
    }
  }

  /** Cases without an answer, answered by the teacher (an AI function, or a model the function is called with). */
  async labelled(cases: readonly Case[]): Promise<Case[]> {
    const p = this.plan;
    const t = p.opts.teacher!;
    const fn = typeof t === "object" && t !== null && "predict" in (t as object) ? t as unknown as Fn : p.fn.using({ lm: t as never });
    const need = cases.filter((c) => !c.outputs);
    this.say(`asking the teacher for ${need.length} answers`);
    const got = await pool(need, 16, async (c) => {
      try {
        return { c, outputs: (await fn.predict(c.inputs as never)).outputs as Rec };
      } catch {
        return { c, outputs: null };
      }
    });
    const answered = new Map(got.filter((g) => g.outputs).map((g) => [g.c, g.outputs]));
    return cases.filter((c) => c.outputs || answered.has(c)).map((c) => (c.outputs ? c : { ...c, outputs: answered.get(c)! }));
  }

  /** `rl` and `distill`: sample, reward (or score each token against the teacher), train. */
  async rollouts(): Promise<void> {
    const p = this.plan;
    const s = p.settings;
    const steps = Number(s["steps"]), per = Number(s["batch"]), perEpoch = Number(s["steps_per_epoch"]), group = Number(s["group"]);
    let teacher: Sampler | null = null;
    let reference: Sampler | null = null;
    if (p.method === "distill") teacher = await this.teacherSampler();
    if (Number(s["kl"]) > 0) reference = await this.run.sampler();          // the starting weights
    if (this.st.step === 0) await this.check(0);
    const cases = uniqueCases(p.train);
    for (let step = this.st.step; step < steps; step++) {
      const epoch = Math.floor(step / perEpoch);
      const k = step % perEpoch;
      const batch = this.order(cases.length, epoch).slice(k * per, (k + 1) * per).map((i) => cases[i]!);
      const sampler = await this.run.sampler();
      const data: ReturnType<typeof wireDatum>[] = [];
      let skipped = 0, rewardSum = 0, rewardN = 0, tokensOut = 0;
      if (p.method === "rl") {
        const got = await this.groups(sampler, batch, Number(s["temperature"]), group);
        const advs: { g: (typeof got)[number]; a: number[] }[] = [];
        for (const g of got) {
          rewardSum += g.rewards.reduce((a, b) => a + b, 0);
          rewardN += g.rewards.length;
          const a = T.advantages(g.rewards, Boolean(s["normalize"]));
          if (!a) { skipped++; continue; }
          advs.push({ g, a });
        }
        let perToken: number[][][] | null = null;
        if (reference && advs.length) {
          const deltas = await pool(advs, 16, async ({ g }) => Promise.all(g.seqs.map(async (q) => {
            const lp = await reference!.logprobs(wireChunks([...g.prompt, { text: q.tokens }], g.pictures));
            const start = chunksLength(g.prompt);
            return q.tokens.map((_, j) => q.logprobs[j]! - (lp[start + j] ?? 0));
          })));
          const flatA = advs.flatMap(({ g, a }) => g.seqs.map((q, i) => q.tokens.map(() => a[i]!)));
          const adj = T.klAdjusted(flatA, deltas.flat(), Number(s["kl"]));
          perToken = [];
          let at = 0;
          for (const { g } of advs) perToken.push(g.seqs.map(() => adj[at++]!));
        }
        advs.forEach(({ g, a }, gi) => g.seqs.forEach((q, i) => {
          tokensOut += q.tokens.length;
          data.push(wireDatum(T.rlDatum(g.prompt, q.tokens, q.logprobs, perToken ? perToken[gi]![i]! : a[i]!), g.pictures));
        }));
      } else {
        const got = await pool(batch, 16, async (c) => {
          const r = this.rendered(c);
          const res = await sampler.sample(wireChunks(r.prompt, r.pictures), { numSamples: group, temperature: Number(s["temperature"]), maxTokens: Number(s["max_tokens"]), stop: [p.student.stopToken], seed: this.seedFor(step, c.key) });
          const start = chunksLength(r.prompt);
          return Promise.all(res.sequences.map(async (q) => {
            const lp = await teacher!.logprobs(wireChunks([...this.teacherPrompt(c, r), { text: q.tokens }], r.pictures));
            const tStart = chunksLength(this.teacherPrompt(c, r));
            const own = q.logprobs ?? q.tokens.map(() => 0);
            const adv = q.tokens.map((_, j) => -(own[j]! - (lp[tStart + j] ?? 0)));     // reverse KL, per token
            rewardSum += adv.reduce((a, b) => a + b, 0) / Math.max(1, adv.length);
            rewardN++;
            tokensOut += q.tokens.length;
            void start;
            return wireDatum(T.rlDatum(r.prompt, q.tokens, own, adv), r.pictures);
          }));
        });
        data.push(...got.flat());
      }
      if (data.length) {
        await both(this.run.forwardBackward(data, String(s["loss"] ?? "importance_sampling")), this.run.optimStep({ learning_rate: Number(s["lr"]) }));
      }
      this.metric({ step: step + 1, reward: rewardN ? rewardSum / rewardN : null, groups_skipped: skipped, answers: data.length, tokens_out: tokensOut });
      await this.afterStep(step + 1);
    }
  }

  /** `distill`: the teacher's sampler (a Tinker base model, or a baked model's weights), refused unless it shares the student's tokenizer. */
  async teacherSampler(): Promise<Sampler> {
    const p = this.plan;
    const t = p.opts.teacher;
    if (t === undefined) throw new BakeError("bake-rows", "method \"distill\" needs a teacher: a Tinker model id, or a baked model on Tinker");
    if (t instanceof BakedModel) {
      const w = t.meta["weights"] as Rec;
      if ((t.meta["template"] as Rec)["tokenizer_sha256"] !== p.student.tokenizerSha256) throw new BakeError("bake-rows", "distill: the teacher's tokenizer is not the student's (on-policy distillation compares token by token); bake with method \"sft\" and teacher: to learn its answers instead");
      return this.tinker.sampler({ path: w["uri"] as string });
    }
    if (typeof t === "string") {
      const ts = await Student.load(t, filesFor(p.opts.studentFiles, t));
      if (ts.tokenizerSha256 !== p.student.tokenizerSha256) throw new BakeError("bake-rows", `distill: ${t}'s tokenizer is not ${p.student.id}'s (on-policy distillation compares token by token); bake with method "sft" and teacher: "${t}" to learn its answers instead`);
      return this.tinker.sampler({ base: t });
    }
    throw new BakeError("bake-rows", "distill: the teacher is a Tinker model id or a baked model on Tinker (an AI function teaches through method \"sft\")");
  }

  /** The teacher's prompt: the student's, or with the fixed inputs shown (context distillation) when the student leaves them out. */
  teacherPrompt(c: Case, r: { prompt: Chunk[] }): Chunk[] {
    const p = this.plan;
    if (!Object.keys(p.entry.fixed).length && !Object.keys(p.entry.derived).length) return r.prompt;
    const full = bakeEntry(p.fn as never, { layout: p.opts.layout, reasoning: p.opts.reasoning });
    const conv = conversation(full, c.inputs, null);
    return p.student.render(conv.messages, conv.pictures).prompt;
  }

  /** `prefer` (DPO, training.md §9). */
  async prefer(): Promise<void> {
    const p = this.plan;
    const s = p.settings;
    const pairs = (p.opts.pairs ?? []).map((pair) => {
      const inputs = checkInputs(p.fn.interface, pair.inputs, p.fn.name);
      const a = conversation(p.entry, inputs, pair.chosen);
      const b = conversation(p.entry, inputs, pair.rejected);
      const ra = p.student.render(a.messages, a.pictures, a.reply);
      const rb = p.student.render(b.messages, b.pictures, b.reply);
      return { chosen: ra, rejected: rb, pictures: a.pictures };
    });
    if (!pairs.length) throw new BakeError("bake-rows", "method \"prefer\" needs pairs: [{ inputs, chosen, rejected }]");
    const sumOf = (lp: number[], d: ReturnType<typeof T.sftDatum>) => lp.reduce((acc, x, i) => acc + x * ((d.lossFnInputs["weights"] as number[])[i]! ? 1 : 0), 0);
    const datum = (r: (typeof pairs)[number]["chosen"]) => T.sftDatum(r.full!, r.replyStart, 1);
    this.say(`the starting weights' log-probabilities of ${pairs.length} pairs`);
    const ref = await this.run.forwardBackward(pairs.flatMap((x) => [wireDatum(datum(x.chosen), x.pictures), wireDatum(datum(x.rejected), x.pictures)]), "cross_entropy", { forwardOnly: true });
    const rho = pairs.map((x, i) => ({ c: sumOf(ref.outputs[2 * i]!.logprobs as number[], datum(x.chosen)), r: sumOf(ref.outputs[2 * i + 1]!.logprobs as number[], datum(x.rejected)) }));
    const steps = Number(s["steps"]), per = Number(s["batch"]), perEpoch = Number(s["steps_per_epoch"]);
    if (this.st.step === 0) await this.check(0);
    for (let step = this.st.step; step < steps; step++) {
      const idx = this.order(pairs.length, Math.floor(step / perEpoch)).slice((step % perEpoch) * per, (step % perEpoch + 1) * per);
      const now = await this.run.forwardBackward(idx.flatMap((i) => [wireDatum(datum(pairs[i]!.chosen), pairs[i]!.pictures), wireDatum(datum(pairs[i]!.rejected), pairs[i]!.pictures)]), "cross_entropy", { forwardOnly: true });
      const terms = idx.map((i, j) => ({ pi_c: sumOf(now.outputs[2 * j]!.logprobs as number[], datum(pairs[i]!.chosen)), rho_c: rho[i]!.c,
        pi_r: sumOf(now.outputs[2 * j + 1]!.logprobs as number[], datum(pairs[i]!.rejected)), rho_r: rho[i]!.r }));
      const w = T.dpo(terms, Number(s["beta"]));
      const data = idx.flatMap((i, j) => [
        wireDatum(T.weightedDatum(pairs[i]!.chosen.full!, pairs[i]!.chosen.replyStart, () => w[j]!.chosen_weight), pairs[i]!.pictures),
        wireDatum(T.weightedDatum(pairs[i]!.rejected.full!, pairs[i]!.rejected.replyStart, () => w[j]!.rejected_weight), pairs[i]!.pictures),
      ]);
      await both(this.run.forwardBackward(data, "cross_entropy"), this.run.optimStep({ learning_rate: Number(s["lr"]) }));
      this.metric({ step: step + 1, loss: w.reduce((a, x) => a + x.loss, 0) / w.length, accuracy: w.filter((x) => x.z > 0).length / w.length });
      await this.afterStep(step + 1);
    }
  }

  /** `decision`: the choices' probabilities at one position, trained by cross-entropy over them (or the Brier score for a row's `target` distribution). */
  async decision(): Promise<void> {
    const p = this.plan;
    const s = p.settings;
    const answerName = p.fn.interface.outputs.at(-1)!.name;
    const choices = decisionChoices(p.fn, p.entry);
    const points = p.train.map((c) => decisionPoint(p.student, p.entry, answerName, choices, c.inputs));
    const targetOf = (c: Case): number[] => {
      const t = c.row["target"];
      if (t && typeof t === "object" && !Array.isArray(t)) return choices.map((ch) => Number((t as Rec)[String(ch)] ?? 0));
      const y = c.outputs?.[answerName];
      return choices.map((ch) => (writeData(toJson(ch)[0]) === writeData(toJson(y)[0]) ? 1 : 0));
    };
    const steps = Number(s["steps"]), per = Number(s["batch"]), perEpoch = Number(s["steps_per_epoch"]);
    if (this.st.step === 0) await this.check(0);
    for (let step = this.st.step; step < steps; step++) {
      const idx = this.order(p.train.length, Math.floor(step / perEpoch)).slice((step % perEpoch) * per, (step % perEpoch + 1) * per);
      const make = (i: number, weights: number[]) => wireDatum(T.decisionDatum([...points[i]!.prefix, { text: [points[i]!.tokens[0]!] }], points[i]!.tokens, weights), points[i]!.pictures);
      const now = await this.run.forwardBackward(idx.map((i) => make(i, points[i]!.tokens.map(() => 0))), "cross_entropy", { forwardOnly: true });
      let loss = 0;
      const data = idx.map((i, j) => {
        const rows = now.outputs[j]!.logprobs as number[][];
        const t = targetOf(p.train[i]!);
        const hard = t.filter((x) => x === 1).length === 1 && t.every((x) => x === 0 || x === 1);
        const d = T.decisionWeights(rows.at(-1)!, hard ? t.indexOf(1) : null, hard ? null : t);
        loss += d.loss;
        return make(i, d.weights);
      });
      await both(this.run.forwardBackward(data, "cross_entropy"), this.run.optimStep({ learning_rate: Number(s["lr"]) }));
      this.metric({ step: step + 1, loss: loss / idx.length });
      await this.afterStep(step + 1);
    }
  }

  // ---- the model

  meta(weights: Rec | null): Rec {
    const p = this.plan;
    const choices = p.method === "decision" ? decisionChoices(p.fn, p.entry) : null;
    const replies = choices && p.train.length ? choiceReplies(p.entry, p.fn.interface.outputs.at(-1)!.name, choices, p.train[0]!.inputs) : null;
    return {
      functai_baked: BAKED_FORMAT, kind: "generative", name: p.name, student: p.student.id, created: new Date().toISOString(),
      functions: [entryJson(p.entry)], weights: weights ?? { form: "remote", service: "tinker", uri: "", base: p.student.id },
      template: p.student.describe(), generation: { max_new_tokens: Number(p.settings["max_tokens"]), max_model_len: Number(p.tokens["longest"]) + Number(p.settings["max_tokens"]) },
      sampling: { temperature: this.st?.best?.temperature ?? 0 },
      training: { method: p.method, where: "tinker", steps: this.st?.step ?? 0, checks: this.st?.checks ?? [], best: this.st?.best ?? null,
        criteria: this.st?.best ? { [this.decider()]: this.st.best.score } : {}, trained_on: p.method === "rl" && p.opts.reward ? [this.rewardName()] : [],
        spent: { estimated: p.dollars["total"] ?? null }, ...(choices ? { decision: { choices, replies } } : {}), run_folder: p.folder },
      hashes: {},
    };
  }

  decider(): string {
    const p = this.plan;
    return p.opts.criteria?.[0]?.name ?? (p.method === "rl" && p.opts.reward ? this.rewardName() : p.opts.metric ? (p.opts.metric.name || "metric") : "exact_match");
  }

  rewardName(): string {
    const r = this.plan.opts.reward!;
    return isGroup(r) ? r.name : typeof r === "function" ? r.name || "reward" : (r as m.Criterion).name;
  }

  async finish(): Promise<BakedModel> {
    const p = this.plan;
    // the state of the weights the model answers with (the best check's), else the last weights'
    let state: string | null = this.st.best?.state ?? null;
    if (!state && p.settings["keep_state"] && !this.st.best?.sampler) state = await this.run.saveState("final");
    for (const old of this.st.states) await this.tinker.deleteCheckpoint(old.path).catch(() => {});
    this.st.states = state ? [{ step: this.st.step, path: state }] : [];
    let sampler = this.st.best?.sampler ?? null;
    if (!sampler) sampler = await this.run.saveForSampler("final");
    const weights = { form: "remote", service: "tinker", uri: sampler, sampler_uri: sampler, ...(state ? { state_uri: state } : {}), base: p.student.id, lora_rank: Number(p.settings["rank"]) };
    const meta = this.meta(weights);
    try {
      (meta["training"] as Rec)["spent"] = spentOf(p.dollars["total"] ?? null, await this.tinker.billed(new Date(this.st.started)), this.st.sessions ?? []);
    } catch {
      /* billing records come later; the estimate stands */
    }
    this.fs.mkdirSync(p.output, { recursive: true });
    this.fs.writeFileSync(this.path.join(p.output, "baked.json"), writeData(meta, 1) + "\n");
    this.say(`saved to ${p.output} (${this.st.best ? `best check: step ${this.st.best.step} at temperature ${this.st.best.temperature}, ${this.decider()} ${this.st.best.score.toFixed(3)}` : "no checks"})`);
    return bakedOnTinker(p.output, { tinker: p.opts.tinker, studentFiles: p.opts.studentFiles, student: p.student });
  }
}

/** Cases with distinct inputs (rl's prompts), first of each. */
function uniqueCases(cases: readonly Case[]): Case[] {
  const seen = new Set<string>();
  return cases.filter((c) => (seen.has(c.key) ? false : (seen.add(c.key), true)));
}

/** A step's two requests (sent together): both settled before either's failure is reported. */
async function both<A, B>(a: Promise<A>, b: Promise<B>): Promise<[A, B]> {
  const [x, y] = await Promise.allSettled([a, b]);
  if (x.status === "rejected") throw x.reason;
  if (y.status === "rejected") throw y.reason;
  return [x.value, y.value];
}

/** Map with at most `n` in flight, results in order. */
async function pool<T, R>(xs: readonly T[], n: number, f: (x: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(xs.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, xs.length) }, async () => {
    for (;;) {
      const i = next++;
      if (i >= xs.length) return;
      out[i] = await f(xs[i]!);
    }
  }));
  return out;
}
