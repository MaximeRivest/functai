/**
 * How often a function is right (contract/scores.md): run it on rows with
 * known answers, score each row, and give the score with its 95% range.
 */

import * as lmcc from "lmcc";
import { newId } from "./calllog.ts";
import type { AnyAIFunction as AIFunction, Expected, Row } from "./fn.ts";
import { withSettings } from "./settings.ts";
import { normalize } from "./text.ts";

type Rec = Record<string, unknown>;

const Z = 1.959964;
const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
  2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042];

function t975(df: number): number {
  if (df <= T975.length) return T975[df - 1]!;
  return Z + (Z ** 3 + Z) / (4 * df) + (5 * Z ** 5 + 16 * Z ** 3 + 3 * Z) / (96 * df ** 2);
}

/** The mean and its 95% interval: Wilson's for right-or-wrong scores, Student's t otherwise. */
export function interval(values: readonly number[]): { mean: number | null; low: number | null; high: number | null } {
  const n = values.length;
  if (n === 0) return { mean: null, low: null, high: null };
  let mean = 0;
  for (const v of values) mean += v;
  mean /= n;
  if (n < 2) return { mean, low: null, high: null };
  if (values.every((v) => v === 0 || v === 1)) {
    const z2 = Z * Z;
    const center = (mean + z2 / (2 * n)) / (1 + z2 / n);
    const half = Z * Math.sqrt(mean * (1 - mean) / n + z2 / (4 * n * n)) / (1 + z2 / n);
    return { mean, low: mean === 0 ? 0 : Math.max(0, center - half), high: mean === 1 ? 1 : Math.min(1, center + half) };
  }
  let ss = 0;
  for (const v of values) ss += (v - mean) ** 2;
  const half = t975(n - 1) * Math.sqrt(ss / (n - 1)) / Math.sqrt(n);
  return { mean, low: mean - half, high: mean + half };
}

function norm(v: unknown): unknown {
  return typeof v === "string" ? normalize(v) : v;
}

function same(a: unknown, b: unknown): boolean {
  a = norm(a);
  b = norm(b);
  if (typeof a === "string" || typeof b === "string") return a === b;
  try {
    return lmcc.jsonEqual(a as lmcc.Json, b as lmcc.Json);
  } catch {
    return a === b;
  }
}

/**
 * The default metric: 1 when every output the answers have a value for
 * equals the prediction's (text compared ignoring case and repeated white
 * space), else 0. With several, each also gets `<name>_match`.
 */
export function exactMatch(answers: Rec, prediction: Rec): Record<string, number> {
  const keys = Object.keys(prediction).filter((k) => k in answers);
  if (!keys.length) throw new Error(`exact_match: the data has no column for any output (${Object.keys(prediction).join(", ")})`);
  const each = Object.fromEntries(keys.map((k) => [k, same(answers[k], prediction[k]) ? 1 : 0]));
  const out: Record<string, number> = { exact_match: Object.values(each).every((v) => v === 1) ? 1 : 0 };
  if (keys.length > 1) for (const [k, v] of Object.entries(each)) out[`${k}_match`] = v;
  return out;
}

/** A metric: `(row, prediction) => number` (0 to 1, or any score). */
export type Metric<R = Rec> = (row: R, prediction: Rec) => number | Promise<number>;

export interface RowResult {
  readonly row: Rec;
  readonly outputs: Rec | null;
  readonly scores: Record<string, number>;
  readonly error: string | null;
  readonly callId: string | null;
}

export interface Summary {
  readonly metric: string;
  readonly mean: number | null;
  readonly low: number | null;
  readonly high: number | null;
  readonly n: number;
  readonly failed: number;
}

/** The result of `evaluate`: a score, its range, and every answer. */
export class Evaluation {
  /** The id of this run: its calls carry it as `caller.evaluation` in the call log. */
  readonly run: string;
  readonly rows: readonly RowResult[];
  readonly metrics: readonly string[];

  constructor(run: string, rows: readonly RowResult[], metrics: readonly string[]) {
    this.run = run;
    this.rows = rows;
    this.metrics = metrics;
  }

  /** Each row's value for a metric (the first by default); a failed row counts 0. */
  scores(metric: string = this.metrics[0]!): number[] {
    return this.rows.map((r) => r.scores[metric] ?? 0);
  }

  /** One line per metric: the mean, its 95% range, how many rows, how many failed. */
  get summary(): Summary[] {
    return this.metrics.map((m) => ({ metric: m, ...interval(this.scores(m)), n: this.rows.length, failed: this.rows.filter((r) => r.error).length }));
  }

  /** The first metric's mean, from 0 to 1. */
  get score(): number | null {
    return interval(this.scores()).mean;
  }

  get low(): number | null {
    return interval(this.scores()).low;
  }

  get high(): number | null {
    return interval(this.scores()).high;
  }

  toString(): string {
    const f = (x: number | null) => (x === null ? "—" : x.toFixed(2));
    return this.summary.map((s) => `${s.metric}: ${f(s.mean)} (95% range ${f(s.low)} to ${f(s.high)}), n=${s.n}${s.failed ? `, ${s.failed} failed` : ""}`).join("\n");
  }
}

export interface EvaluateOptions<F = AIFunction, R = Rec> {
  /** Where the right answers are: a column of the rows for the answer, or `{output: column}`. Default: columns named like the outputs. */
  expected?: Expected<F, R>;
  /** A metric, or several by name. Default: exact_match. */
  metric?: Metric<R> | Record<string, Metric<R>>;
  /** Calls in flight at once (default 8). */
  concurrency?: number;
}

/**
 * Run `fn` on every row and score it. A row's inputs are its columns named
 * like the function's inputs; the right answers are the columns named like
 * its outputs (or `expected`). Calls are logged with `caller.evaluation`.
 */
export async function evaluate<F extends AIFunction, R extends Row<F>>(fn: F, rows: readonly R[], opts: EvaluateOptions<F, R> = {}): Promise<Evaluation> {
  const run = newId();
  const inputNames = fn.definition.inputs.map((f) => f.name);
  const outputNames = fn.definition.outputs.map((f) => f.name);
  const mapping: Record<string, string> = typeof opts.expected === "string" ? { [fn.answerName]: opts.expected }
    : (opts.expected as Record<string, string> | undefined) ?? Object.fromEntries(outputNames.filter((n) => rows.some((r) => n in r)).map((n) => [n, n]));
  const custom = typeof opts.metric === "function" ? { [opts.metric.name || "metric"]: opts.metric } : opts.metric;
  if (!custom && !Object.keys(mapping).length) {
    throw new Error(`evaluate: the rows have no column for any output (${outputNames.join(", ")}); pass expected or a metric`);
  }
  const results: RowResult[] = new Array(rows.length);
  let next = 0;
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= rows.length) return;
      const row = rows[i]!;
      const inputs = Object.fromEntries(inputNames.filter((n) => n in row).map((n) => [n, row[n]]));
      try {
        const pred = await withSettings({ caller: { evaluation: run } }, () => fn.predict(inputs));
        const outputs = pred.outputs as Rec;
        let scores: Record<string, number>;
        if (custom) {
          scores = {};
          for (const [name, m] of Object.entries(custom)) scores[name] = Number(await m(row, outputs));
        } else {
          const answers = Object.fromEntries(Object.entries(mapping).map(([out, col]) => [out, row[col]]));
          scores = exactMatch(answers, Object.fromEntries(Object.keys(mapping).map((k) => [k, outputs[k]])));
        }
        results[i] = { row, outputs, scores, error: null, callId: pred.callId };
      } catch (err) {
        results[i] = { row, outputs: null, scores: {}, error: `${(err as Error).name}: ${(err as Error).message}`, callId: null };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(opts.concurrency ?? 8, rows.length || 1)) }, worker));
  const metrics = custom ? Object.keys(custom)
    : ["exact_match", ...(Object.keys(mapping).length > 1 ? Object.keys(mapping).map((k) => `${k}_match`) : [])];
  return new Evaluation(run, results, metrics);
}
