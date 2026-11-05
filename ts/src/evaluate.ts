/**
 * How often a function is right (contract/scores.md): run it on rows with
 * known answers, score each row, and give the score with its 95% range.
 * Each call says which case it answered (contract/measures.md, *Results*),
 * so later evaluations can reuse it and comparisons pair by case.
 */

import * as lmcc from "lmcc";
import * as calllog from "./calllog.ts";
import { newId } from "./calllog.ts";
import type { AnyAIFunction as AIFunction, Expected, Row } from "./fn.ts";
import { effective, withSettings } from "./settings.ts";
import { normalize } from "./text.ts";
import { replaying } from "./conversations.ts";
import { Progress, showProgress } from "./progress.ts";
import type { Interface } from "./interface.ts";
import type { Stream } from "./stream.ts";
import { getOwn, toJson } from "./values.ts";
import * as m from "./measures.ts";

type Rec = Record<string, unknown>;

const Z = 1.959964;
const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
  2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042];

export function t975(df: number): number {
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
  const keys = Object.keys(prediction).filter((k) => Object.hasOwn(answers, k));
  if (!keys.length) throw new Error(`exact_match: the data has no column for any output (${Object.keys(prediction).join(", ")})`);
  const each = Object.fromEntries(keys.map((k) => [k, same(answers[k], prediction[k]) ? 1 : 0]));
  const out: Record<string, number> = { exact_match: Object.values(each).every((v) => v === 1) ? 1 : 0 };
  if (keys.length > 1) for (const [k, v] of Object.entries(each)) out[`${k}_match`] = v;
  return out;
}

/** A metric: `(row, prediction) => number` (0 to 1, or any score). */
export type Metric<R = Rec> = (row: R, prediction: Rec) => number | Promise<number>;

/** One run: one repeat of one row. */
export interface RowResult {
  readonly row: Rec;
  readonly outputs: Rec | null;
  /** Each metric's value; a criterion without a value (no item judged) is absent. */
  readonly scores: Record<string, number>;
  readonly error: string | null;
  /** The call that answered: its trace in the call log. */
  readonly callId: string | null;
  /** The row's position in the data. */
  readonly example: number;
  /** The row's case key (measures.md). */
  readonly caseKey: string;
  /** Which time the row was asked, from 1. */
  readonly repeat: number;
  /** Taken from the log instead of asked (`reuse`). */
  readonly reused: boolean;
}

export interface Summary {
  readonly metric: string;
  readonly mean: number | null;
  readonly low: number | null;
  readonly high: number | null;
  /** Cases with a value. */
  readonly n: number;
  readonly failed: number;
}

/** The result of `evaluate`: a score, its range over cases, and every answer. */
export class Evaluation {
  /** The id of this run: its calls carry it as `caller.evaluation` in the call log. */
  readonly run: string;
  /** Every run: `repeats` per row, in row order. */
  readonly rows: readonly RowResult[];
  readonly metrics: readonly string[];
  /** Each row's case key (`caseKey`), in data order. */
  readonly cases: readonly string[];
  /** A criterion's data by its metric's name: its values count as measures.md says (no value is none, not 0). */
  readonly kinds: Readonly<Record<string, Rec>>;

  constructor(run: string, rows: readonly RowResult[], metrics: readonly string[], cases?: readonly string[], kinds: Readonly<Record<string, Rec>> = {}) {
    this.run = run;
    this.rows = rows;
    this.metrics = metrics;
    this.cases = cases ?? rows.map((r) => r.caseKey);
    this.kinds = kinds;
  }

  /** A metric's value per data row: the mean of its runs (a failed run counts 0, except for a criterion's run with no value). */
  private perRow(metric: string): (number | null)[] {
    const legacy = !Object.hasOwn(this.kinds, metric);
    const acc: number[][] = this.cases.map(() => []);
    for (const r of this.rows) {
      const v = Object.hasOwn(r.scores, metric) ? r.scores[metric]! : legacy ? 0 : null;
      if (v !== null) acc[r.example]!.push(v);
    }
    return acc.map((a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null));
  }

  /** Each data row's value for a metric (the first by default): the mean of its repeats; a failed row counts 0. */
  scores(metric: string = this.metrics[0]!): number[] {
    return this.perRow(metric).map((v) => v ?? 0);
  }

  /** A metric's value per case: rows of one case, and repeats, averaged. */
  byCase(metric: string = this.metrics[0]!): Map<string, number> {
    const acc = new Map<string, number[]>();
    this.perRow(metric).forEach((v, i) => {
      if (v === null) return;
      const k = this.cases[i]!;
      acc.set(k, [...(acc.get(k) ?? []), v]);
    });
    return new Map([...acc].map(([k, vs]) => [k, vs.reduce((s, x) => s + x, 0) / vs.length] as const));
  }

  /** Whether a metric's values lie between 0 and 1 (exact match, a bounded criterion). */
  bounded(metric: string): boolean {
    const kind = this.kinds[metric];
    return kind ? m.bounded(kind) : metric === "exact_match" || metric.endsWith("_match");
  }

  /** Each run's call id, in order. */
  get calls(): (string | null)[] {
    return this.rows.map((r) => r.callId);
  }

  private stats(metric: string) {
    const per = this.byCase(metric);
    const keys = [...per.keys()].sort();
    const iv = interval(keys.map((k) => per.get(k)!));
    if (iv.low !== null && iv.high !== null && Object.hasOwn(this.kinds, metric) && this.bounded(metric)) {
      return { ...iv, low: Math.max(0, iv.low), high: Math.min(1, iv.high), n: keys.length };
    }
    return { ...iv, n: keys.length };
  }

  /** One line per metric: the mean over cases, its 95% range, how many cases, how many runs failed. */
  get summary(): Summary[] {
    return this.metrics.map((metric) => {
      const s = this.stats(metric);
      return { metric, mean: s.mean, low: s.low, high: s.high, n: s.n, failed: this.rows.filter((r) => r.error && !Object.hasOwn(r.scores, metric)).length };
    });
  }

  /** The first metric's mean over cases, from 0 to 1. */
  get score(): number | null {
    return this.metrics.length ? this.stats(this.metrics[0]!).mean : null;
  }

  get low(): number | null {
    return this.metrics.length ? this.stats(this.metrics[0]!).low : null;
  }

  get high(): number | null {
    return this.metrics.length ? this.stats(this.metrics[0]!).high : null;
  }

  toString(): string {
    const f = (x: number | null) => (x === null ? "—" : x.toFixed(2));
    const reused = this.rows.filter((r) => r.reused).length;
    return this.summary.map((s) => `${s.metric}: ${f(s.mean)} (95% range ${f(s.low)} to ${f(s.high)}), n=${s.n}${s.failed ? `, ${s.failed} failed` : ""}${reused ? `, ${reused} reused` : ""}`).join("\n");
  }
}

export interface EvaluateOptions<F = AIFunction, R = Rec> {
  /** Where the right answers are: a column of the rows for the answer, or `{output: column}`. Default: columns named like the outputs. */
  expected?: Expected<F, R>;
  /** A metric, or several by name. Default: exact_match. */
  metric?: Metric<R> | Record<string, Metric<R>>;
  /**
   * Questions with a judge (measures.ts `criterion(...)` with `judge`): each
   * becomes a metric named like it, its ratings are kept in the call log
   * (naming the judge by its setup or code), and a judgement already there
   * for the same call and judge is reused.
   */
  criteria?: readonly m.Criterion[];
  /** How many times each row is asked (default 1); a repeat after the first is another replicate. */
  repeats?: number;
  /**
   * Take results already in this log (a store, a folder, records, or true:
   * the call log): calls of the same program on the same setup and case that
   * ended without an error, oldest first, count as the first repeats. This
   * evaluation's calls and ratings go there too.
   */
  reuse?: calllog.CallStore | string | readonly Rec[] | true;
  /** Calls in flight at once (default 8). */
  concurrency?: number;
  /** A line on stderr, updated as rows finish (default: on in a terminal). */
  progress?: boolean;
}

/** What `evaluate` runs: an AI function, a module, or a program served elsewhere (`remote`). */
type Evaluable = AIFunction | { readonly name: string; readonly interface: Interface; readonly setup?: string; readonly module?: string; stream(input: unknown, options?: Rec): Stream };

function reuseRecords(reuse: EvaluateOptions["reuse"]): [Rec[], calllog.CallStore | string | null] {
  if (reuse === true) {
    const [c, r] = calllog.read(calllog.folderOf(effective({}).logCalls ?? true));
    return [[...c, ...r], null];
  }
  if (Array.isArray(reuse)) return [[...(reuse as readonly Rec[])], null];
  if (calllog.isStore(reuse)) return [(reuse.read() as Rec[]).map((r) => ({ ...r })), reuse];
  const [c, r] = calllog.read(reuse as string);
  return [[...c, ...r], reuse as string];
}

/**
 * Run `fn` on every row and score it. A row's inputs are its columns named
 * like the function's inputs; the right answers are the columns named like
 * its outputs (or `expected`). Calls are logged with `caller.evaluation`,
 * and each says its case (`case`, `repeat`).
 */
export async function evaluate<F extends Evaluable, R extends Rec>(fn: F, rows: readonly (F extends AIFunction ? R & Row<F> : R)[],
  opts: EvaluateOptions<F, R> = {}): Promise<Evaluation> {
  const run = newId();
  const iface = fn.interface;
  const inputNames = iface.inputs.map((f) => f.name);
  const outputNames = iface.outputs.map((f) => f.name);
  const answerName = outputNames[outputNames.length - 1]!;
  const isAi = typeof (fn as { predict?: unknown }).predict === "function";
  const repeats = opts.repeats ?? 1;
  if (!Number.isInteger(repeats) || repeats < 1) throw new RangeError(`repeats is a whole number of at least 1, not ${repeats}`);
  const crits = (opts.criteria ?? []).map((c) => {
    if (c.kind === "choice") throw new TypeError(`${c.name} is a choice between two answers: judgePairs`);
    if (!c.judge) throw new TypeError(`${c.name} has no judge: evaluate scores with judges (criterion(..., { judge })); people's ratings are read by scoreboard`);
    return c;
  });
  /** One row's call: every output, and the call's id. A row from a conversation is asked with its earlier turns. */
  const ask = (row: Rec, inputs: Rec): Promise<{ outputs: Rec; callId: string | null; tokens: number }> => replaying(row, fn, async () => {
    if (isAi) {
      const pred = await (fn as AIFunction).predict(inputs as never);
      return { outputs: pred.outputs as Rec, callId: pred.callId, tokens: pred.usage["totalTokens"] ?? 0 };
    }
    const s = (fn as { stream(i: unknown): Stream }).stream(inputs);
    const value = await s.result;
    return { outputs: outputNames.length === 1 ? { [answerName]: value } : value as Rec, callId: s.callId, tokens: 0 };
  });
  const mapping: Record<string, string> = typeof opts.expected === "string" ? { [answerName]: opts.expected }
    : (opts.expected as Record<string, string> | undefined) ?? Object.fromEntries(outputNames.filter((n) => rows.some((r) => Object.hasOwn(r, n))).map((n) => [n, n]));
  const custom = typeof opts.metric === "function" ? { [opts.metric.name || "metric"]: opts.metric } : opts.metric;
  const useExact = !custom && Object.keys(mapping).length > 0;
  if (!custom && !useExact && !crits.length) {
    throw new Error(`evaluate: the rows have no column for any output (${outputNames.join(", ")}); pass expected, a metric or criteria`);
  }
  const keys = rows.map((r) => m.rowKey(r as Rec, inputNames));

  // results already in the log: the program's own setup, on the same case, ended well, oldest first
  const reused = new Map<string, Rec[]>();
  const known = new Map<string, unknown>();
  let store: calllog.CallStore | string | null = null;
  if (opts.reuse !== undefined) {
    const [found, where] = reuseRecords(opts.reuse);
    store = where;
    const setup = (fn as { setup?: string }).setup;
    const key: m.ProgramKey = { name: fn.name, ...((fn as { module?: string }).module !== undefined ? { module: (fn as { module?: string }).module! } : {}) };
    for (const c of m.programCalls(found, key)) {
      const p = (c["program"] ?? {}) as Rec;
      if (p["setup"] !== setup || (c["error"] !== null && c["error"] !== undefined) || c["content"] !== true || typeof c["outputs"] !== "object" || !c["outputs"]) continue;
      const k = m.callKey(c);
      if (k !== null) reused.set(k, [...(reused.get(k) ?? []), c]);
    }
    const ratings = found.filter((r) => "functai_rating" in r);
    for (const c of crits) {
      const ident = m.judgeIdentity(c.judge);
      const jid = String(ident["setup"] ?? ident["code"]);
      for (const j of m.counting(ratings, m.criterionData(c))) {
        if (j.source[0] === "judge" && j.source[1] === jid && j.calls.length === 1) known.set(JSON.stringify([j.calls[0], j.item, jid]), j.value);
      }
    }
  }
  const jobs: { example: number; repeat: number; prior: Rec | null }[] = [];
  keys.forEach((k, i) => {
    const have = reused.get(k) ?? [];
    for (let rep = 1; rep <= repeats; rep++) jobs.push({ example: i, repeat: rep, prior: have[rep - 1] ?? null });
  });

  const results: RowResult[] = new Array(jobs.length);
  const toAsk = jobs.filter((j) => j.prior === null).length;
  const meter = showProgress(opts.progress) && toAsk ? new Progress(toAsk, fn.name) : null;
  const kept: Rec[] = [];

  /** A criterion's value for one run: its judge asked (or its rating reused), its ratings kept. */
  const judgeRun = async (c: m.Criterion, row: Rec, outputs: Rec | null, failed: boolean, callId: string | null): Promise<number | null> => {
    const data = m.criterionData(c);
    if (failed || !outputs) return c.items === undefined || c.aggregate === "sum" ? 0 : null;
    const outJson = toJson(outputs)[0] as Rec;
    const inputs: Rec = {};
    for (const n of inputNames) if (Object.hasOwn(row, n)) inputs[n] = toJson(row[n])[0];
    const exp: Rec = {};
    for (const [o, col] of Object.entries(mapping)) if (Object.hasOwn(row, col)) exp[o] = toJson(row[col])[0];
    const base: Rec = { ...inputs, inputs, outputs: outJson, answer: outJson[answerName],
      expected: Object.keys(exp).length <= 1 ? exp[answerName] : exp };
    const who = { judge: m.judgeIdentity(c.judge) };
    const jid = String(who.judge["setup"] ?? who.judge["code"]);
    const targets: [string | null, unknown][] = c.items === undefined ? [[null, undefined]]
      : ((): [string, unknown][] => {
        const lst = m.pointer(outJson, c.items!);
        return Array.isArray(lst) ? lst.map((it, i) => [`${c.items}/${i}`, it]) : [];
      })();
    const values: number[] = [];
    for (const [item, it] of targets) {
      const priorKey = JSON.stringify([callId, item ?? "", jid]);
      let value: unknown;
      if (callId !== null && known.has(priorKey)) value = known.get(priorKey);
      else {
        const [got, jc] = await m.askJudge(c.judge!, item === null ? base : { ...base, item: it });
        value = m.judgedValue(data, got);
        if (callId !== null) kept.push(m.ratingRecord(callId, data, value, { who, item, origin: "judge", judgeCall: jc }));
      }
      const x = m.numberOf({ kind: c.kind, value, answers: c.answers as Record<string, number | null> | undefined, scale: c.scale as [number, number] | undefined });
      if (x !== null) values.push(x);
    }
    if (c.items === undefined) return values.length ? values[0]! : null;
    if (!values.length) return c.aggregate === "sum" && !targets.length ? 0 : null;
    const share = values.reduce((s, x) => s + x, 0) / values.length;
    return c.aggregate === "sum" ? share * targets.length : share;
  };

  let next = 0;
  const worker = async () => {
    for (;;) {
      const n = next++;
      if (n >= jobs.length) return;
      const job = jobs[n]!;
      const row = rows[job.example]! as Rec;
      const caseKey = keys[job.example]!;
      const inputs = Object.fromEntries(inputNames.filter((x) => Object.hasOwn(row, x)).map((x) => [x, row[x]]));
      let outputs: Rec | null = null, callId: string | null = null, error: string | null = null;
      if (job.prior) {
        outputs = job.prior["outputs"] as Rec;
        callId = String(job.prior["id"]);
      } else {
        const mark: calllog.Mark = { case: caseKey, repeat: job.repeat, call: null };
        try {
          const got = await calllog.MARK.run(mark, () => withSettings({ caller: { evaluation: run }, ...(job.repeat > 1 ? { replicate: job.repeat - 1 } : {}) }, () => ask(row, inputs)));
          outputs = got.outputs;
          callId = got.callId ?? mark.call;
          meter?.step(false, got.tokens);
        } catch (err) {
          meter?.step(true, 0);
          const e = err as { name?: unknown; message?: unknown } | null | undefined;     // a call may fail with anything, undefined included
          error = `${typeof e?.name === "string" ? e.name : "Error"}: ${typeof e?.message === "string" ? e.message : String(err)}`;
          callId = mark.call;
        }
      }
      const scores: Record<string, number> = {};
      const note = (what: string, err: unknown) => {
        const said = `${what}: ${(err as Error)?.name ?? "Error"}: ${(err as Error)?.message ?? String(err)}`;
        error = error ? `${error}; ${said}` : said;
      };
      if (error === null && outputs) {
        if (custom) {
          for (const [name, metric] of Object.entries(custom)) {
            try {
              scores[name] = Number(await metric(row as R, outputs));
            } catch (err) {
              note(`metric ${name}`, err);
            }
          }
        } else if (useExact) {
          const answers = Object.fromEntries(Object.entries(mapping).map(([out, col]) => [out, getOwn(row, col)]));
          Object.assign(scores, exactMatch(answers, Object.fromEntries(Object.keys(mapping).map((k) => [k, getOwn(outputs!, k)]))));
        }
      }
      for (const c of crits) {
        try {
          const v = await judgeRun(c, row, outputs, error !== null && !outputs, callId);
          if (v !== null) scores[c.name] = v;
        } catch (err) {
          note(`criterion ${c.name}`, err);
        }
      }
      results[n] = { row, outputs: error !== null && !outputs ? null : outputs, scores, error, callId, example: job.example, caseKey, repeat: job.repeat, reused: job.prior !== null };
    }
  };
  const go = () => Promise.all(Array.from({ length: Math.max(1, Math.min(opts.concurrency ?? 8, jobs.length || 1)) }, worker));
  if (store !== null) await withSettings({ logCalls: store as never }, go);
  else await go();
  if (kept.length) {
    try {
      for (const r of kept) await calllog.writeRating(r, store, effective({}));
    } catch {
      // no call log here: the values stand, unkept
    }
  }
  const metrics = [...(custom ? Object.keys(custom) : useExact ? ["exact_match", ...(Object.keys(mapping).length > 1 ? Object.keys(mapping).map((k) => `${k}_match`) : [])] : []),
    ...crits.map((c) => c.name)];
  return new Evaluation(run, results, metrics, keys, Object.fromEntries(crits.map((c) => [c.name, m.criterionData(c)])));
}
