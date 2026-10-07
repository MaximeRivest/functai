/**
 * Comparing two evaluations of the same rows, row by row. Pairing the rows
 * detects a real change with far fewer examples than two separate scores
 * would: a row both versions got right says nothing, a row only one got right
 * says a lot.
 */

import { t975, type Evaluation } from "./evaluate.ts";
import { writeData } from "./values.ts";

/** One metric both evaluations have: the means, the paired difference with its 95% interval, and how many rows changed. */
export interface Comparison {
  readonly metric: string;
  readonly before: number;
  readonly after: number;
  /** `after - before`, the mean over rows. */
  readonly diff: number;
  /** Its 95% interval (a paired t interval); null with one row. When it includes 0, the change could be luck. */
  readonly low: number | null;
  readonly high: number | null;
  readonly better: number;
  readonly worse: number;
  readonly same: number;
  readonly n: number;
}

/**
 * Compare two evaluations of the same rows (two versions of a prompt, two
 * models), row by row: one line per metric both have.
 *
 * ```ts
 * compare(await evaluate(category, rows), await evaluate(categoryV2, rows));
 * // [{ metric: "exact_match", before: 0.6, after: 0.8, diff: 0.2, low: -0.05, high: 0.45, better: 2, worse: 0, same: 8, n: 10 }]
 * ```
 */
/** What differs between two rows: the columns only one has, else the first that differs. */
function howRowsDiffer(a: unknown, b: unknown): string {
  if (typeof a !== "object" || a === null || typeof b !== "object" || b === null) return "the rows are not the same";
  const ka = Object.keys(a), kb = Object.keys(b);
  const onlyA = ka.filter((k) => !kb.includes(k)), onlyB = kb.filter((k) => !ka.includes(k));
  const q = (ks: string[]) => ks.map((k) => `'${k}'`).join(", ");
  if (onlyA.length || onlyB.length) {
    return [onlyA.length ? `column ${q(onlyA)} only in the first` : "", onlyB.length ? `column ${q(onlyB)} only in the second` : ""]
      .filter(Boolean).join("; ");
  }
  const differ = ka.find((k) => writeData((a as Record<string, unknown>)[k]) !== writeData((b as Record<string, unknown>)[k]));
  return differ !== undefined ? `column '${differ}' differs` : "the rows are not the same";
}

export function compare(before: Evaluation, after: Evaluation): Comparison[] {
  if (before.rows.length !== after.rows.length) throw new RangeError(`compare needs the same examples: ${before.rows.length} rows vs ${after.rows.length}`);
  before.rows.forEach((a, i) => {
    if (writeData(a.row) !== writeData(after.rows[i]!.row)) {
      throw new RangeError(`compare needs the same examples in the same order; row ${i} differs: ${howRowsDiffer(a.row, after.rows[i]!.row)}. Evaluate both versions on the very same table, and put the change in the function (a second AI function, or another model), not in the data.`);
    }
  });
  const shared = before.metrics.filter((m) => after.metrics.includes(m));
  if (!shared.length) throw new RangeError(`no metric in common: ${before.metrics.join(", ")} vs ${after.metrics.join(", ")}`);
  return shared.map((metric) => {
    const a = before.scores(metric);
    const b = after.scores(metric);
    const d = b.map((y, i) => y - a[i]!);
    const n = d.length;
    const mean = d.reduce((s, v) => s + v, 0) / n;
    let low: number | null = null;
    let high: number | null = null;
    if (n >= 2) {
      const sd = Math.sqrt(d.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1));
      const half = t975(n - 1) * sd / Math.sqrt(n);
      low = mean - half;
      high = mean + half;
    }
    return {
      metric, before: a.reduce((s, v) => s + v, 0) / n, after: b.reduce((s, v) => s + v, 0) / n, diff: mean, low, high,
      better: d.filter((v) => v > 0).length, worse: d.filter((v) => v < 0).length, same: d.filter((v) => v === 0).length, n,
    };
  });
}
