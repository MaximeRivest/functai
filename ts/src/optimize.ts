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

import { exactMatch, type Metric } from "./evaluate.ts";
import type { AnyAIFunction as AIFunction, Demo } from "./fn.ts";
import { withSettings } from "./settings.ts";
import { newId } from "./calllog.ts";

type Rec = Record<string, unknown>;

function random(seed: number): () => number {
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

function labeled(fn: AIFunction, row: Rec): Demo | null {
  const inputs = Object.fromEntries(fn.definition.inputs.filter((f) => f.name in row).map((f) => [f.name, row[f.name]]));
  const outputs = Object.fromEntries(fn.definition.outputs.filter((f) => f.name in row).map((f) => [f.name, row[f.name]]));
  return Object.keys(outputs).length ? { inputs, outputs } : null;
}

/** Up to `k` rows with known answers become worked examples (a seeded random sample, or the first `k`). */
export function labeledFewShot<F extends AIFunction>(fn: F, rows: readonly Rec[], opts: { k?: number; sample?: boolean; seed?: number } = {}): F {
  const k = opts.k ?? 16;
  const chosen = opts.sample === false ? rows.slice(0, k) : sample(rows, k, opts.seed ?? 0);
  const copy = fn.using({}) as F;
  copy.demos = chosen.map((r) => labeled(fn, r)).filter((d): d is Demo => d !== null);
  return copy;
}

export interface BootstrapOptions {
  /** Which runs are good: `(row, outputs) => score`; default exact_match against the row's answers. */
  metric?: Metric;
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
 * Run the function (or a teacher model) on rows with known answers; the runs
 * the metric accepts become worked examples, whole turns included (reasoning,
 * tool calls). Labeled rows fill the rest, up to `maxLabeled`.
 */
export async function bootstrapFewShot<F extends AIFunction>(fn: F, rows: readonly Rec[], opts: BootstrapOptions = {}): Promise<F> {
  const maxBoot = opts.maxBootstrapped ?? 4;
  const maxLabeled = opts.maxLabeled ?? 16;
  const runner = opts.teacher ? fn.using({ lm: opts.teacher }) : fn;
  const outputs = fn.definition.outputs.map((f) => f.name);
  const metric: Metric = opts.metric ?? ((row, pred) => {
    const answers = Object.fromEntries(outputs.filter((o) => o in row).map((o) => [o, row[o]]));
    return exactMatch(answers, Object.fromEntries(Object.keys(answers).map((k) => [k, pred[k]])))["exact_match"]!;
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
      const inputs = Object.fromEntries(fn.definition.inputs.filter((f) => f.name in row).map((f) => [f.name, row[f.name]]));
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
  const fill = room > 0 ? sample(rest, room, opts.seed ?? 0).map((r) => labeled(fn, r)).filter((d): d is Demo => d !== null) : [];
  const copy = fn.using({}) as F;
  copy.demos = [...boot, ...fill] as never;
  return copy;
}
