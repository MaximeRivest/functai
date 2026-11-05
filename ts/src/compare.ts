/**
 * Comparing two evaluations case by case (contract/measures.md, *Is B
 * better than A*). Pairing the cases detects a real change with far fewer
 * examples than two separate scores would: a case both versions got right
 * says nothing, a case only one got right says a lot. Cases are paired by
 * key (their inputs), whatever the rows' order.
 */

import type { Evaluation } from "./evaluate.ts";
import { paired, type Outcome } from "./measures.ts";

/** One metric both evaluations have: the means on shared cases, the paired difference with its 95% interval, and what it says. */
export interface Comparison {
  readonly metric: string;
  readonly before: number;
  readonly after: number;
  /** `after - before`, the mean over shared cases. */
  readonly diff: number;
  /** Its 95% interval (MOVER-Wilson for right-or-wrong values, paired t otherwise); null with one case. */
  readonly low: number | null;
  readonly high: number | null;
  readonly better: number;
  readonly worse: number;
  readonly same: number;
  readonly n: number;
  readonly method: "mover-wilson" | "paired-t" | null;
  /** `better`, `worse`, and with a margin `same`, `not_worse`, `not_better`; else `unclear`. */
  readonly outcome: Outcome;
  /** The fewest cases whose removal would erase the difference. */
  readonly fragility: number;
  /** Cases only one evaluation answered. */
  readonly onlyBefore: number;
  readonly onlyAfter: number;
}

/**
 * Compare two evaluations case by case (two versions of a prompt, two
 * models): one line per metric both have.
 *
 * ```ts
 * compare(await evaluate(category, rows), await evaluate(categoryV2, rows), { margin: 0.05 });
 * // [{ metric: "exact_match", before: 0.6, after: 0.8, diff: 0.2, low: -0.07, high: 0.43, outcome: "unclear", … }]
 * ```
 */
export function compare(before: Evaluation, after: Evaluation, opts: { margin?: number | null } = {}): Comparison[] {
  const shared = before.metrics.filter((m) => after.metrics.includes(m));
  if (!shared.length) throw new RangeError(`no metric in common: ${before.metrics.join(", ")} vs ${after.metrics.join(", ")}`);
  return shared.map((metric) => {
    const a = before.byCase(metric), b = after.byCase(metric);
    const keys = [...a.keys()].filter((k) => b.has(k)).sort();
    if (!keys.length) {
      throw new RangeError(`compare needs cases both answered; ${metric} has none in common (${a.size} cases before, ${b.size} after, none with the same inputs)`);
    }
    const xs = keys.map((k) => a.get(k)!), ys = keys.map((k) => b.get(k)!);
    const p = paired(xs, ys, opts.margin, before.bounded(metric) && after.bounded(metric));
    return {
      metric, before: xs.reduce((s, v) => s + v, 0) / xs.length, after: ys.reduce((s, v) => s + v, 0) / ys.length, diff: p.diff!, low: p.low, high: p.high,
      better: p.better, worse: p.worse, same: p.same, n: p.n, method: p.method, outcome: p.outcome, fragility: p.fragility,
      onlyBefore: [...a.keys()].filter((k) => !b.has(k)).length, onlyAfter: [...b.keys()].filter((k) => !a.has(k)).length,
    };
  });
}
