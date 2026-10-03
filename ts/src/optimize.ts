/**
 * Improving a function: choosing its worked examples. Each returns an
 * improved copy; the function you pass is unchanged. Improving never edits
 * the fields or the layout, only what the version counts as the program's
 * state (the instruction and the demos), so the copy has a new version.
 *
 * The contract fixes what improving means, not its random choices
 * (design/01-many-languages.md, decision 3): a seed makes a run repeatable
 * here, not equal to Python's run with the same seed.
 */

import { evaluate, exactMatch, type Metric } from "./evaluate.ts";
import { ai } from "./fn.ts";
import { t } from "./shapes.ts";
import { inContext } from "./conversations.ts";
import { warnOnce } from "./calllog.ts";
import { writeData } from "./values.ts";
import type { AnyAIFunction as AIFunction, Demo, Expected, Row } from "./fn.ts";
import { withSettings } from "./settings.ts";
import { newId } from "./calllog.ts";

type Rec = Record<string, unknown>;

export function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample<T>(items: readonly T[], k: number, seed: number): T[] {
  const rng = random(seed);
  const pool = [...items];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j]!, pool[i]!];
  }
  return pool.slice(0, Math.min(k, pool.length));
}

/**
 * The rows that can be worked examples: a row answered after earlier turns
 * (`rated()` on a conversation gives its `earlier`) is measured, never shown
 * as a worked example (a worked example is one turn placed before the
 * question); how many were skipped is said once.
 */
function standalone<R>(fn: AIFunction, rows: readonly R[]): R[] {
  const kept = rows.filter((r) => !inContext(r as Rec));
  if (kept.length < rows.length) {
    warnOnce(`in-context:${fn.name}:${rows.length - kept.length}`, `${fn.name}: ${rows.length - kept.length} of ${rows.length} rows were answered after earlier turns; they are measured, never shown as worked examples`);
  }
  return kept;
}

/** Where each output's right answer is in a row: `expected`, or the columns named like the outputs. */
export function answerColumns(fn: AIFunction, expected: unknown): Record<string, string> {
  if (typeof expected === "string") return { [fn.answerName]: expected };
  if (expected && typeof expected === "object") return { ...(expected as Record<string, string>) };
  return Object.fromEntries(fn.definition.outputs.map((f) => [f.name, f.name]));
}

function labeled(fn: AIFunction, row: Rec, columns: Record<string, string>): Demo | null {
  const inputs = Object.fromEntries(fn.definition.inputs.filter((f) => Object.hasOwn(row, f.name)).map((f) => [f.name, row[f.name]]));
  const outputs = Object.fromEntries(Object.entries(columns).filter(([, col]) => Object.hasOwn(row, col)).map(([out, col]) => [out, row[col]]));
  return Object.keys(outputs).length ? { inputs, outputs } : null;
}

export interface LabeledOptions<F = AIFunction, R = Rec> {
  /** How many rows at most (default 16). */
  k?: number;
  /** A seeded random sample (default), or the first `k`. */
  sample?: boolean;
  seed?: number;
  /** Where the right answers are: a column for the answer, or `{output: column}`. Default: columns named like the outputs. */
  expected?: Expected<F, R>;
}

/** An improved copy: up to `k` rows with known answers become worked examples. The function is unchanged. */
export function labeledFewShot<F extends AIFunction, R extends Row<F>>(fn: F, rows: readonly R[], opts: LabeledOptions<F, R> = {}): F {
  rows = standalone(fn, rows);
  const k = opts.k ?? 16;
  const columns = answerColumns(fn, opts.expected);
  const chosen = opts.sample === false ? rows.slice(0, k) : sample(rows, k, opts.seed ?? 0);
  const copy = fn.using({}) as F;
  copy.demos = chosen.map((r) => labeled(fn, r, columns)).filter((d): d is Demo => d !== null);
  return copy;
}

export interface BootstrapOptions<F = AIFunction, R = Rec> {
  /** Where the right answers are, as in `evaluate`. */
  expected?: Expected<F, R>;
  /** Which runs are good: `(row, outputs) => score`; default exact_match against the row's answers. */
  metric?: Metric<R>;
  /** A run counts when its score is at least this (default: any score above 0). */
  threshold?: number;
  maxBootstrapped?: number;
  maxLabeled?: number;
  /** A stronger model to write the examples (`"gpt-4.1"`); default the function's own. */
  teacher?: string;
  seed?: number;
  /** Runs in flight at once (default 4). */
  concurrency?: number;
}

/**
 * An improved copy: the function (or a teacher model) runs on rows with known
 * answers, and the runs the metric accepts become worked examples, whole turns
 * included (reasoning, tool calls). Labeled rows fill the rest, up to
 * `maxLabeled`. The function is unchanged.
 */
export async function bootstrapFewShot<F extends AIFunction, R extends Row<F>>(fn: F, rows: readonly R[], opts: BootstrapOptions<F, R> = {}): Promise<F> {
  rows = standalone(fn, rows);
  const maxBoot = opts.maxBootstrapped ?? 4;
  const maxLabeled = opts.maxLabeled ?? 16;
  const runner = opts.teacher ? fn.using({ lm: opts.teacher }) : fn;
  const columns = answerColumns(fn, opts.expected);
  const metric: Metric<R> = opts.metric ?? ((row, pred) => {
    const answers = Object.fromEntries(Object.entries(columns).filter(([, col]) => Object.hasOwn(row, col)).map(([out, col]) => [out, row[col]]));
    return exactMatch(answers, Object.fromEntries(Object.keys(answers).map((k) => [k, Object.hasOwn(pred, k) ? pred[k] : undefined])))["exact_match"]!;
  });
  const passes = (score: number) => (opts.threshold !== undefined ? score >= opts.threshold : score > 0);
  const boot: Rec[] = [];
  const used = new Set<number>();
  const id = newId();
  let next = 0;
  const worker = async () => {
    while (boot.length < maxBoot) {
      const i = next++;
      if (i >= rows.length) return;
      const row = rows[i]!;
      const inputs = Object.fromEntries(fn.definition.inputs.filter((f) => Object.hasOwn(row, f.name)).map((f) => [f.name, row[f.name]]));
      try {
        const pred = await withSettings({ caller: { optimization: id } }, () => runner.predict(inputs));
        if (passes(Number(await metric(row, pred.outputs as Rec))) && boot.length < maxBoot) {
          boot.push(pred.turn.toJSON() as unknown as Rec);
          used.add(i);
        }
      } catch {
        // a failed run is not an example
      }
    }
  };
  if (maxBoot > 0) await Promise.all(Array.from({ length: Math.max(1, opts.concurrency ?? 4) }, worker));
  const room = maxLabeled - boot.length;
  const rest = rows.filter((_, i) => !used.has(i));
  const fill = room > 0 ? sample(rest, room, opts.seed ?? 0).map((r) => labeled(fn, r, columns)).filter((d): d is Demo => d !== null) : [];
  const copy = fn.using({}) as F;
  copy.demos = [...boot, ...fill] as never;
  return copy;
}

// ------------------------------------------------------------------ searches

/** The first metric's mean score of a candidate on rows (a failed row counts 0). */
async function meanScore<F extends AIFunction, R extends Row<F>>(fn: F, rows: readonly R[], opts: { expected?: Expected<F, R>; metric?: Metric<R>; concurrency?: number }): Promise<number> {
  const ev = await evaluate(fn, rows, { ...(opts.expected ? { expected: opts.expected } : {}), ...(opts.metric ? { metric: opts.metric } : {}), concurrency: opts.concurrency ?? 8, progress: false });
  return ev.score ?? 0;
}

export interface RandomSearchOptions<F = AIFunction, R = Rec> extends BootstrapOptions<F, R> {
  /** Bootstrapped candidates besides zero-shot, labeled and bootstrapped (default 8). */
  candidates?: number;
  /** Rows to score candidates on (default: the rows). */
  valset?: readonly R[];
  /** Stop once a candidate scores this much. */
  stopAtScore?: number;
}

/**
 * Try several sets of worked examples and keep the one that scores best on
 * the validation rows: none, labeled only, bootstrapped, and bootstrapped
 * from shuffled rows with random sizes, each scored on `valset` (default: the
 * rows). Python's `BootstrapFewShotWithRandomSearch`. Returns the best copy
 * and every candidate (`{ candidate, demos, score }`).
 */
export async function randomSearch<F extends AIFunction, R extends Row<F>>(fn: F, rows: readonly R[], opts: RandomSearchOptions<F, R> = {}): Promise<{ fn: F; candidates: { candidate: string; demos: number; score: number }[] }> {
  const val = opts.valset ?? rows;
  const seed = opts.seed ?? 0;
  const maxBoot = opts.maxBootstrapped ?? 4;
  const scoring = { ...(opts.expected ? { expected: opts.expected } : {}), ...(opts.metric ? { metric: opts.metric } : {}), concurrency: opts.concurrency };
  const candidates: { candidate: string; demos: number; score: number }[] = [];
  let best: [number, F] = [-1, fn];
  for (let k = -3; k < (opts.candidates ?? 8); k++) {
    let label: string;
    let copy: F;
    if (k === -3) {
      label = "zero-shot";
      copy = fn.using({}) as F;
      copy.demos = [];
    } else if (k === -2) {
      label = "labeled";
      copy = labeledFewShot(fn, rows, { k: opts.maxLabeled ?? 16, seed, ...(opts.expected ? { expected: opts.expected } : {}) });
    } else {
      let shuffled = [...rows];
      let size = maxBoot;
      if (k >= 0) {
        const rng = random(seed + k + 1);
        shuffled = sample(shuffled, shuffled.length, seed + k + 1);
        size = 1 + Math.floor(rng() * Math.max(1, maxBoot));
      }
      label = k === -1 ? "bootstrapped" : `bootstrapped (seed ${k}, ${size} demos)`;
      copy = await bootstrapFewShot(fn, shuffled, { ...opts, maxBootstrapped: size, seed: seed + Math.max(0, k) });
    }
    const score = await meanScore(copy, val, scoring);
    candidates.push({ candidate: label, demos: copy.demos.length, score });
    if (score > best[0]) best = [score, copy];
    if (opts.stopAtScore !== undefined && score >= opts.stopAtScore) break;
  }
  return { fn: best[1], candidates };
}

const TIPS = [
  "", "Be concise and direct.", "Be precise: the task is high-stakes and mistakes are costly.", "Describe the expected output format exactly.",
  "Spell out the steps to follow before answering.", "Name the common mistakes on this task and how to avoid them.",
  "Give the model a helpful persona suited to the task.", "Consider edge cases and unusual inputs.",
];

let proposer: AIFunction | null = null;
function proposeInstruction(): AIFunction {
  proposer ??= ai("propose_instruction", {
    description: "Propose a new instruction (a system prompt) for an AI function that will make it score higher on its task. Use the "
      + "signature and the examples to understand the task. Make it different from the previous proposals, and follow the tip. Keep the exact "
      + "input and output names. Reply with the instruction text only.",
    input: { function_name: t.string(), signature: t.string(), current_instruction: t.string(), examples: t.string(),
      previous_proposals: t.list(t.string()), tip: t.string() },
    output: t.string(), definedIn: "functai.meta",
  }) as unknown as AIFunction;
  return proposer;
}

export interface InstructionSearchOptions<F = AIFunction, R = Rec> extends BootstrapOptions<F, R> {
  /** Instructions (the current one and proposals) and demo sets to try (default 6). */
  candidates?: number;
  /** Minibatch evaluations (default 12). */
  trials?: number;
  /** Rows per minibatch (default 20). */
  minibatch?: number;
  /** The model that writes instructions (default: the configured one). */
  promptLm?: string;
  /** Rows to score on (default: the rows). */
  valset?: readonly R[];
  /** How many top combinations are scored on every validation row (default 3). */
  finalists?: number;
}

/**
 * Search instructions written by a model, with demo sets, and keep the best
 * (MIPRO-style, as Python's `InstructionSearch`): instruction candidates (the
 * current one plus proposals written by `promptLm` from the signature and a
 * few rows) × demo sets (bootstrapped, unless both demo limits are 0),
 * searched over `trials` minibatch evaluations; the top combinations are then
 * scored on every validation row and the best wins. The search is random
 * with greedy refinement, not Bayesian. Returns the best copy and every trial.
 */
export async function instructionSearch<F extends AIFunction, R extends Row<F>>(fn: F, rows: readonly R[], opts: InstructionSearchOptions<F, R> = {}): Promise<{ fn: F; trials: Rec[] }> {
  const seed = opts.seed ?? 0;
  const rng = random(seed);
  const n = Math.max(1, opts.candidates ?? 6);
  const val = opts.valset ?? rows;
  const minibatch = opts.minibatch ?? 20;
  const scoring = { ...(opts.expected ? { expected: opts.expected } : {}), ...(opts.metric ? { metric: opts.metric } : {}), concurrency: opts.concurrency };
  const inputs = fn.definition.inputs.map((f) => f.name);
  const examplesText = (list: readonly Rec[]) => list.slice(0, 5).map((r) => writeData({
    inputs: Object.fromEntries(Object.entries(r).filter(([k]) => inputs.includes(k))), outputs: Object.fromEntries(Object.entries(r).filter(([k]) => !inputs.includes(k))),
  })).join("\n");
  const signatureText = `${fn.name}(${inputs.join(", ")}) -> ${fn.definition.outputs.map((f) => f.name).join(", ")}`;
  const instructions: (string | null)[] = [fn.state().instructions];
  const proposals: string[] = [];
  const id = newId();
  for (let i = 0; i < n - 1; i++) {
    const shuffled = sample(rows as readonly Rec[], rows.length, seed + i + 1);
    const writer = opts.promptLm ? proposeInstruction().using({ lm: opts.promptLm, temperature: 1 }) : proposeInstruction().using({ temperature: 1 });
    const text = String(await withSettings({ caller: { optimization: id } }, () => writer({
      function_name: fn.name, signature: signatureText, current_instruction: fn.instructions, examples: examplesText(shuffled) || "(none)",
      previous_proposals: proposals, tip: TIPS[i % TIPS.length] || "(no tip)",
    } as never)) ?? "").trim();
    if (text && !proposals.includes(text)) {
      proposals.push(text);
      instructions.push(text);
    }
  }
  const demoSets: Rec[][] = [fn.state().demos as Rec[]];
  if ((opts.maxBootstrapped ?? 4) > 0 || (opts.maxLabeled ?? 4) > 0) {
    for (let k = 0; k < n - 1; k++) {
      const shuffled = sample(rows, rows.length, seed + k);
      const boot = await bootstrapFewShot(fn, shuffled, { ...opts, maxLabeled: opts.maxLabeled ?? 4, seed: seed + k });
      demoSets.push(boot.state().demos as Rec[]);
    }
  }
  const candidate = (combo: readonly number[]): F => {
    const copy = fn.using({}) as F;
    copy.loadState({ instructions: instructions[combo[0]!]!, demos: demoSets[combo[1]!]! as never });
    return copy;
  };
  const sizes = [instructions.length, demoSets.length];
  const scores = new Map<string, number[]>();
  const trials: Rec[] = [];
  const mean = (k: string) => scores.get(k)!.reduce((a, b) => a + b, 0) / scores.get(k)!.length;
  const trialsN = Math.max(1, opts.trials ?? 12);
  for (let tr = 0; tr < trialsN; tr++) {
    let combo: number[];
    if (tr === 0) combo = sizes.map(() => 0);
    else if (scores.size && tr >= Math.floor(trialsN / 2) && rng() < 0.6) {
      const best = [...scores.keys()].sort((a, b) => mean(b) - mean(a))[0]!.split(",").map(Number);
      const j = Math.floor(rng() * sizes.length);
      combo = best.map((v, i) => (i === j ? Math.floor(rng() * sizes[i]!) : v));
    } else combo = sizes.map((m) => Math.floor(rng() * m));
    const batch = val.length <= minibatch ? val : sample(val, minibatch, seed + 100 + tr);
    const score = await meanScore(candidate(combo), batch, scoring);
    const key = combo.join(",");
    scores.set(key, [...(scores.get(key) ?? []), score]);
    trials.push({ trial: tr, combo, minibatch_score: score });
  }
  const ranked = [...scores.keys()].sort((a, b) => mean(b) - mean(a));
  const finalists = ranked.slice(0, Math.max(1, opts.finalists ?? 3));
  let bestKey = finalists[0]!;
  if (val.length > minibatch) {
    const full = new Map<string, number>();
    for (const k of finalists) full.set(k, await meanScore(candidate(k.split(",").map(Number)), val, scoring));
    bestKey = [...full.keys()].sort((a, b) => full.get(b)! - full.get(a)!)[0]!;
    for (const tr of trials) {
      const k = (tr["combo"] as number[]).join(",");
      if (full.has(k)) tr["full_score"] = full.get(k);
    }
  }
  return { fn: candidate(bestKey.split(",").map(Number)), trials };
}
