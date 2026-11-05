/**
 * Measuring programs: cases, results, ratings and honest numbers
 * (contract/measures.md, design/20). A **case** is the inputs a program can
 * be asked again, named by its key; a **result** is one call of one setup on
 * one case (a call record carrying `case`); a **rating** judges a result, an
 * item of it, or two results side by side, by a person, an account, a judge
 * program or code. Everything here is statistics over them, the same in
 * every language that measures: every range is over cases, right-or-wrong
 * comparisons use the MOVER-Wilson interval, and a judge is combined with a
 * random draw of people's ratings by prediction-powered inference (PPI++).
 *
 * ```ts
 * scoreboard(study, [realProblem], { records });          // per setup: people, people drawn, each judge, combined
 * compareSetups(study, old.setup, next.setup, realProblem, { source: professor, margin: 0.05, records });
 * rank(study, preferred, { records });                    // Bradley–Terry over every setup, with fragility
 * ```
 */

import * as lmcc from "lmcc";
import * as calllog from "./calllog.ts";
import { effective } from "./settings.ts";
import { toJson } from "./values.ts";
import type { Prediction } from "./engine.ts";

type Rec = Record<string, unknown>;

const Z = 1.959964;
const CONTEXT = ["earlier", "sections", "helpers"] as const;
const KINDS = ["verdict", "label", "score", "choice"] as const;
const RANK_STEPS = 20;

/** A measure refused what it was given (`code`: `case-split`, …). */
export class MeasureError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(message);
    this.name = "MeasureError";
    this.code = code;
  }
}

const mean = (xs: readonly number[]): number => {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
};
const variance = (xs: readonly number[]): number => {
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) ** 2;
  return s / (xs.length - 1);
};
const sum = (xs: readonly number[]): number => {
  let s = 0;
  for (const x of xs) s += x;
  return s;
};
/** Code point order, as Python's sorted() orders text. */
const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const sorted = (xs: Iterable<string>): string[] => [...xs].sort(cmp);
const tuple = (xs: readonly unknown[]): string => JSON.stringify(xs);

// ------------------------------------------------------------------ criteria

/** How a criterion's answers are written: a verdict, a label, a score, or a pick between two answers. */
export type CriterionKind = (typeof KINDS)[number];

/** What a judge is given, by name: the case's inputs, and these. */
export interface JudgeArgs extends Rec {
  inputs: Rec;
  answer?: unknown;
  outputs?: Rec;
  expected?: unknown;
  item?: unknown;
  first?: unknown;
  second?: unknown;
}

/**
 * Who judges automatically: an AI function, a module, or code given one
 * object of what it may use (`({ item, url }) => …`). Its answer, or its
 * answer's `value` field, is the judgement: true or false (or "right",
 * "wrong") for a verdict, an answer for a label, a number for a score,
 * "first", "second" or "tie" for a choice.
 */
export type Judge = ((args: JudgeArgs) => unknown) | { readonly name: string; readonly setup: string; readonly version: string; readonly interface: { inputs: readonly { name: string; optional?: boolean }[] } };

/** A question an answer is judged by (contract/measures.md, *Criteria*). */
export interface Criterion {
  readonly name: string;
  readonly kind: CriterionKind;
  /** A label's answers and the number each counts for (from 0 to 1), or null: it counts for nothing ("unsure"). */
  readonly answers?: Readonly<Record<string, number | null>>;
  /** A score's `[lowest, highest]`. */
  readonly scale?: readonly [number, number];
  /** A JSON Pointer to a list in the answer (`"/insights"`): each item is judged. */
  readonly items?: string;
  /** How a result's items add up: the share of good items ("mean"), or how many ("sum"). */
  readonly aggregate?: "mean" | "sum";
  readonly question?: string;
  readonly judge?: Judge;
}

/**
 * A criterion, checked: `criterion("real_problem", { kind: "label", items: "/insights",
 * answers: { yes: 1, no: 0, unsure: null }, question: "Is this a real problem?" })`.
 */
export function criterion(name: string, opts: Omit<Criterion, "name"> | { kind?: CriterionKind } & Omit<Partial<Criterion>, "name"> = {}): Criterion {
  if (typeof name !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new TypeError(`a criterion's name is letters, digits and _, not ${JSON.stringify(name)}`);
  const kind = (opts as Partial<Criterion>).kind ?? "verdict";
  if (!(KINDS as readonly string[]).includes(kind)) throw new TypeError(`a criterion's kind is one of ${KINDS.join(", ")}, not ${JSON.stringify(kind)}`);
  const o = opts as Partial<Criterion>;
  if (kind === "label") {
    if (!o.answers || !Object.keys(o.answers).length) throw new TypeError(`${name}: a label needs its answers, each the number it counts for (or null)`);
    for (const [k, v] of Object.entries(o.answers)) {
      if (v !== null && !(typeof v === "number" && v >= 0 && v <= 1)) throw new TypeError(`${name}: answer ${JSON.stringify(k)} counts for a number from 0 to 1, or null`);
    }
  } else if (o.answers !== undefined) throw new TypeError(`${name}: only a label has answers`);
  if (o.scale !== undefined && (kind !== "score" || o.scale.length !== 2 || !(o.scale[0] < o.scale[1]))) {
    throw new TypeError(`${name}: a scale is a score's [lowest, highest], lowest below highest`);
  }
  if (o.aggregate !== undefined && o.aggregate !== "mean" && o.aggregate !== "sum") throw new TypeError(`${name}: aggregate is "mean" or "sum"`);
  if (o.items !== undefined) {
    if (kind === "choice") throw new TypeError(`${name}: a choice is between two answers, not items`);
    if (typeof o.items !== "string" || !o.items.startsWith("/")) throw new TypeError(`${name}: items is a JSON Pointer to a list in the answer, like "/insights"`);
  }
  return Object.freeze({ name, kind, ...o, ...(o.items !== undefined ? { aggregate: o.aggregate ?? "mean" } : {}) }) as Criterion;
}

/** "Is the answer right?": what format-1 ratings mean. */
export const correct: Criterion = criterion("correct");
/** "Which answer is better?": a pick between two. */
export const preferred: Criterion = criterion("preferred", { kind: "choice", question: "Which answer is better?" });

/** A criterion as the contract writes it (no judge). */
export function criterionData(c: Criterion | Rec | string): Rec {
  if (typeof c === "string") return { name: c, kind: "verdict" };
  const x = c as Criterion;
  const out: Rec = { name: x.name, kind: x.kind };
  if (x.items !== undefined) {
    out["items"] = x.items;
    out["aggregate"] = x.aggregate ?? "mean";
  }
  if (x.answers !== undefined) out["answers"] = { ...x.answers };
  if (x.scale !== undefined) out["scale"] = [...x.scale];
  if (x.question) out["question"] = x.question;
  return out;
}

/** A criterion's values lie between 0 and 1 (not a count of items, nor a score without a scale). */
export function bounded(c: Rec): boolean {
  if (c["items"] !== undefined && c["items"] !== null && (c["aggregate"] ?? "mean") === "sum") return false;
  return c["kind"] !== "score" || (c["scale"] !== undefined && c["scale"] !== null);
}

// ------------------------------------------------------------------ case keys

/** The key of inputs given as JSON, and a conversation's context. */
export function caseKeyOf(inputs: Rec, context: Rec = {}): string {
  const doc: Rec = { inputs };
  for (const k of CONTEXT) {
    const v = context[k];
    if (v === undefined || v === null) continue;
    if (Array.isArray(v) && !v.length) continue;
    if (typeof v === "object" && !Array.isArray(v) && !Object.keys(v as Rec).length) continue;
    doc[k] = v;
  }
  return lmcc.sha256(doc as lmcc.Json);
}

/** A row's case key for a program whose inputs are `inputs`, its values written as a call record writes them. */
export function rowKey(row: Rec, inputs: readonly string[]): string {
  const ins: Rec = {};
  for (const n of inputs) if (Object.hasOwn(row, n)) ins[n] = toJson(row[n])[0];
  const ctx: Rec = {};
  for (const k of CONTEXT) if (Object.hasOwn(row, k) && !inputs.includes(k)) ctx[k] = toJson(row[k])[0];
  return caseKeyOf(ins, ctx);
}

function inputsKept(call: Rec): boolean {
  const described = ((call["described"] ?? {}) as Rec)["inputs"] as unknown[] | undefined;
  if (described?.length) return false;
  if (call["content"] === true) return Object.hasOwn(call, "inputs");
  const om = call["omitted"] as { inputs?: unknown[] } | undefined;
  return call["functai_call"] === 2 && om !== undefined && Array.isArray(om.inputs) && om.inputs.length === 0;
}

/** A call record's case: its `case`, else its recorded inputs' key when it saw nothing; else null. */
export function callKey(call: Rec): string | null {
  if (typeof call["case"] === "string") return call["case"];
  const saw = call["saw"] as unknown[] | undefined;
  if (call["functai_call"] !== 2 || (saw && saw.length) || !inputsKept(call)) return null;
  return caseKeyOf((call["inputs"] ?? {}) as Rec);
}

type Named = { readonly interface: { readonly inputs: readonly { readonly name: string }[] } };
const inputNames = (program: Named | readonly string[]): string[] =>
  Array.isArray(program) ? [...program] : (program as Named).interface.inputs.map((f) => f.name);

/**
 * The key of the case a row is for a program: its values of the program's
 * inputs (and a conversation's earlier turns), nothing else; a call
 * `evaluate` makes on the row carries it.
 */
export function caseKey(program: Named | readonly string[], row: Rec): string {
  return rowKey(row, inputNames(program));
}

// ------------------------------------------------------------------ held out

/**
 * Whether a row is for training or held out, the same in every language:
 * its own `split`, else by the hash of its key (its conversation's id, so a
 * conversation is on one side) against the held-out `share`.
 */
export function splitOf(row: Rec, program?: Named | readonly string[] | null, opts: { share?: number; salt?: string } = {}): "train" | "held_out" {
  const names = program ? inputNames(program) : [];
  if (Object.hasOwn(row, "split") && !names.includes("split")) {
    const s = row["split"];
    if (s !== "train" && s !== "held_out") throw new MeasureError("case-split", `split is "train" or "held_out", not ${JSON.stringify(s)}`);
    return s;
  }
  const share = opts.share ?? 0.2;
  if (!(share >= 0 && share <= 1)) throw new MeasureError("case-split", `the held-out share is between 0 and 1, not ${share}`);
  if (!program) throw new TypeError("splitOf needs the program (or its input names) for a row without a split");
  const conv = names.includes("conversation") ? null : row["conversation"];
  const group = typeof conv === "string" && conv ? conv : rowKey(row, names);
  const h = lmcc.sha256Hex(`functai-split:${opts.salt ?? ""}:${group}`);
  return parseInt(h.slice(0, 13), 16) / 2 ** 52 < share ? "held_out" : "train";
}

/** How many rows say they are held out. */
export function heldOut(rows: readonly unknown[]): number {
  return rows.filter((r) => typeof r === "object" && r !== null && (r as Rec)["split"] === "held_out").length;
}

// ------------------------------------------------------------------ random draws

type CallLike = string | { callId: string | null } | { id: string } | Rec;

function idOf(c: CallLike): string {
  if (typeof c === "string") return c;
  const r = c as Rec;
  if ("callId" in r) {
    if (r["callId"] === null || r["callId"] === undefined) throw new TypeError("this prediction has no call id (it was not made by calling a program)");
    return String(r["callId"]);
  }
  return String(r["id"] ?? r["call"]);
}

/**
 * A random draw of `n` ids, the same in every language: ranked by the hash
 * of the seed and each id; uniform, reproducible, stable. Rate the drawn
 * calls with `sample: seed`.
 */
export function draw(ids: Iterable<CallLike>, n: number, seed: string): string[] {
  const plain = [...new Set([...ids].map(idOf))];
  const h = new Map(plain.map((i) => [i, lmcc.sha256Hex(`functai-draw:${seed}:${i}`)] as const));
  plain.sort((a, b) => cmp(h.get(a)!, h.get(b)!) || cmp(a, b));
  return plain.slice(0, Math.max(0, Math.trunc(n)));
}

// ------------------------------------------------------------------ reading ratings

/** One rating, read: format 1 or 2 as one shape. */
export interface Judgement {
  id: string;
  at: string;
  calls: string[];
  item: string;
  criterion: string;
  kind: string;
  value: unknown;
  /** `[kind, who]`: `person` and a name, `account` and the rating's id, `judge` and its setup or code. */
  source: [string, string];
  blind: boolean;
  sample: string | null;
  answers?: Record<string, number | null>;
  scale?: [number, number];
  judge?: string;
}

function sourceOf(r: Rec): [string, string] {
  if (typeof r["by"] === "string" && r["by"]) return ["person", r["by"]];
  const j = r["judge"] as Rec | undefined;
  if (j && typeof j === "object") return ["judge", String(j["setup"] ?? j["code"])];
  return ["account", String(r["id"])];
}

/** A rating record of format 1 or 2 as one shape; null for another format. */
export function judgement(r: Rec): Judgement | null {
  const fmt = r["functai_rating"];
  if (fmt === 1) {
    return { id: String(r["id"]), at: String(r["at"]), calls: [String(r["call"])], item: "", criterion: "correct", kind: "verdict",
      value: r["verdict"] ?? null, source: sourceOf(r), blind: r["blind"] === true, sample: (r["sample"] as string | undefined) ?? null };
  }
  if (fmt === 2) {
    if (typeof r["criterion"] !== "string" || typeof r["kind"] !== "string" || !("value" in r) || !("calls" in r || "call" in r)) return null;
    const out: Judgement = {
      id: String(r["id"]), at: String(r["at"]), calls: Array.isArray(r["calls"]) ? (r["calls"] as unknown[]).map(String) : [String(r["call"])],
      item: typeof r["item"] === "string" ? r["item"] : "", criterion: r["criterion"], kind: r["kind"], value: r["value"] ?? null,
      source: sourceOf(r), blind: r["blind"] === true, sample: (r["sample"] as string | undefined) ?? null,
    };
    if (r["kind"] === "label") out.answers = { ...((r["answers"] ?? {}) as Record<string, number | null>) };
    if (r["kind"] === "score" && Array.isArray(r["scale"])) out.scale = [Number((r["scale"] as unknown[])[0]), Number((r["scale"] as unknown[])[1])];
    const j = r["judge"] as Rec | undefined;
    if (j && typeof j === "object") out.judge = String(j["name"]);
    return out;
  }
  return null;
}

const later = (a: Judgement, b: Judgement) => (a.at !== b.at ? a.at > b.at : a.id > b.id);

/** The judgements that count for a criterion (measures.md, *What counts*), by (at, id). */
export function counting(ratings: Iterable<Rec>, crit: Rec): Judgement[] {
  const latest = new Map<string, Judgement>();
  for (const r of ratings) {
    const j = judgement(r);
    if (!j || j.criterion !== crit["name"] || j.kind !== crit["kind"]) continue;
    const [kind, who] = j.source;
    const calls = kind === "person" ? sorted(j.calls) : j.calls;
    const key = kind === "account" ? tuple([kind, who]) : tuple([kind, who, calls, j.item]);
    const had = latest.get(key);
    if (!had || later(j, had)) latest.set(key, j);
  }
  return [...latest.values()].filter((j) => j.value !== null).sort((a, b) => cmp(a.at, b.at) || cmp(a.id, b.id));
}

/** A single answer's judgement as a number, or null (it counts for nothing). */
export function numberOf(j: Pick<Judgement, "kind" | "value" | "answers" | "scale">): number | null {
  const v = j.value;
  if (j.kind === "verdict") return v === "right" ? 1 : v === "wrong" ? 0 : null;
  if (j.kind === "label") {
    const x = typeof v === "string" && j.answers && Object.hasOwn(j.answers, v) ? j.answers[v] : null;
    return x === null || x === undefined ? null : x;
  }
  if (j.kind === "score") {
    if (typeof v !== "number") return null;
    return j.scale ? (v - j.scale[0]) / (j.scale[1] - j.scale[0]) : v;
  }
  return null;
}

const matches = (j: Judgement, source: string, drawn: boolean) => {
  const [kind, who] = j.source;
  if (source === "people") return (kind === "person" || kind === "account") && (!drawn || Boolean(j.sample));
  return kind === "judge" && who === source;
};

// ------------------------------------------------------------------ results and values

/** Which calls are a program's: its name, and its module and file when given. */
export interface ProgramKey {
  name: string;
  module?: string | null;
  file?: string | null;
}

export function programCalls(records: Iterable<Rec>, program: ProgramKey): Rec[] {
  const best = new Map<string, Rec>();
  for (const r of records) {
    if (r["functai_call"] !== 1 && r["functai_call"] !== 2) continue;
    const p = (r["program"] ?? {}) as Rec;
    if (p["name"] !== program.name) continue;
    if (program.module !== undefined && program.module !== null && p["module"] !== program.module) continue;
    if (program.file !== undefined && program.file !== null && p["file"] !== program.file) continue;
    const id = String(r["id"]);
    const had = best.get(id);
    if (!had || Number(r["writer"] ?? 1) > Number(had["writer"] ?? 1)) best.set(id, r);
  }
  return [...best.values()].sort((a, b) => cmp(String(a["started"] ?? ""), String(b["started"] ?? "")) || cmp(String(a["id"]), String(b["id"])));
}

interface Result { call: Rec; group: string | null; case: string; failed: boolean }

export function results(records: Iterable<Rec>, program: ProgramKey, by = "setup"): Result[] {
  const out: Result[] = [];
  for (const c of programCalls(records, program)) {
    const k = callKey(c);
    if (k === null) continue;
    const g = ((c["program"] ?? {}) as Rec)[by];
    out.push({ call: c, group: g === undefined || g === null ? null : String(g), case: k, failed: c["error"] !== null && c["error"] !== undefined });
  }
  return out;
}

const MISSING = Symbol("missing");

/** RFC 6901 into JSON data. */
export function pointer(value: unknown, path: string): unknown {
  if (path === "") return value;
  for (const raw of path.split("/").slice(1)) {
    const tok = raw.replace(/~1/g, "/").replace(/~0/g, "~");
    if (Array.isArray(value)) {
      if (!/^(0|[1-9][0-9]*)$/.test(tok) || Number(tok) >= value.length) return MISSING;
      value = value[Number(tok)];
    } else if (typeof value === "object" && value !== null && Object.hasOwn(value, tok)) value = (value as Rec)[tok];
    else return MISSING;
  }
  return value;
}

const targetKey = (calls: readonly string[], item: string) => tuple([calls, item]);

function resultValue(res: Result, crit: Rec, byTarget: Map<string, number[]>): number | null {
  const items = crit["items"] as string | undefined;
  const agg = (crit["aggregate"] as string | undefined) ?? "mean";
  const cid = String(res.call["id"]);
  if (res.failed) return items === undefined || agg === "sum" ? 0 : null;
  if (items === undefined) {
    const vs = byTarget.get(targetKey([cid], ""));
    return vs && vs.length ? mean(vs) : null;
  }
  const lst = pointer(res.call["outputs"], items);
  if (!Array.isArray(lst)) return agg === "sum" ? 0 : null;
  const per: number[] = [];
  for (let i = 0; i < lst.length; i++) {
    const vs = byTarget.get(targetKey([cid], `${items}/${i}`));
    if (vs && vs.length) per.push(mean(vs));
  }
  if (!per.length) return agg === "sum" && !lst.length ? 0 : null;
  const share = mean(per);
  return agg === "sum" ? share * lst.length : share;
}

function judged(ratings: Iterable<Rec>, crit: Rec, source: string, drawn: boolean): Map<string, number[]> {
  const out = new Map<string, number[]>();
  for (const j of counting(ratings, crit)) {
    if (j.calls.length !== 1 || !matches(j, source, drawn)) continue;
    const x = numberOf(j);
    if (x === null) continue;
    const k = targetKey(j.calls, j.item);
    const list = out.get(k) ?? [];
    list.push(x);
    out.set(k, list);
  }
  return out;
}

const splitRecords = (records: Iterable<Rec>): [Rec[], Rec[]] => {
  const calls: Rec[] = [];
  const ratings: Rec[] = [];
  for (const r of records) {
    if ("functai_rating" in r) ratings.push(r);
    else if ("functai_call" in r) calls.push(r);
  }
  return [calls, ratings];
};

/** Group → (case → value), in order of first result. */
export type CaseValues = Map<string | null, Map<string, number>>;

/** Per result its value, and per group its value by case (measures.md, *Values*). */
export function valuesOf(records: readonly Rec[], program: ProgramKey, crit: Rec, source: string,
  opts: { by?: string; drawn?: boolean } = {}): { results: { call: string; group: string | null; case: string; value: number | null }[]; cases: CaseValues } {
  const [calls, ratings] = splitRecords(records);
  const byTarget = judged(ratings, crit, source, opts.drawn ?? false);
  const per: { call: string; group: string | null; case: string; value: number | null }[] = [];
  const table = new Map<string | null, Map<string, number[]>>();
  for (const res of results(calls, program, opts.by ?? "setup")) {
    const v = resultValue(res, crit, byTarget);
    per.push({ call: String(res.call["id"]), group: res.group, case: res.case, value: v });
    if (v === null) continue;
    const g = table.get(res.group) ?? new Map<string, number[]>();
    const list = g.get(res.case) ?? [];
    list.push(v);
    g.set(res.case, list);
    table.set(res.group, g);
  }
  const cases: CaseValues = new Map();
  for (const [g, cs] of table) cases.set(g, new Map([...cs].map(([k, vs]) => [k, mean(vs)] as const)));
  return { results: per, cases };
}

/** Case → B's share side by side (measures.md, *Side by side*), in key order. */
export function preferencesOf(records: readonly Rec[], program: ProgramKey, crit: Rec, source: string, a: string | null, b: string | null,
  opts: { by?: string; drawn?: boolean } = {}): Map<string, number> {
  const [calls, ratings] = splitRecords(records);
  const picks = new Map<string, Judgement[]>();
  for (const j of counting(ratings, crit)) {
    if (j.calls.length === 2 && matches(j, source, opts.drawn ?? false)) {
      const k = tuple(sorted(j.calls));
      picks.set(k, [...(picks.get(k) ?? []), j]);
    }
  }
  const res = results(calls, program, opts.by ?? "setup");
  const ra = res.filter((r) => r.group === a);
  const rb = res.filter((r) => r.group === b);
  const out = new Map<string, number[]>();
  for (const x of ra) {
    for (const y of rb) {
      if (x.case !== y.case) continue;
      const xid = String(x.call["id"]), yid = String(y.call["id"]);
      let v: number;
      if (x.failed || y.failed) v = x.failed && y.failed ? 0.5 : x.failed ? 1 : 0;
      else {
        const js = picks.get(tuple(sorted([xid, yid])));
        if (!js || !js.length) continue;
        v = mean(js.map((j) => (j.value === yid ? 1 : j.value === xid ? 0 : 0.5)));
      }
      out.set(x.case, [...(out.get(x.case) ?? []), v]);
    }
  }
  return new Map(sorted(out.keys()).map((k) => [k, mean(out.get(k)!)] as const));
}

// ------------------------------------------------------------------ the statistics

const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
  2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042];
const t975 = (df: number): number => (df <= T975.length ? T975[df - 1]! : Z + (Z ** 3 + Z) / (4 * df) + (5 * Z ** 5 + 16 * Z ** 3 + 3 * Z) / (96 * df ** 2));

/** scores.md's interval: Wilson's for 0-or-1 values, Student's t otherwise. */
function interval(values: readonly number[]): { mean: number | null; low: number | null; high: number | null } {
  const n = values.length;
  if (!n) return { mean: null, low: null, high: null };
  const m = mean(values);
  if (n < 2) return { mean: m, low: null, high: null };
  if (values.every((v) => v === 0 || v === 1)) {
    const [low, high] = wilson(sum(values), n);
    return { mean: m, low, high };
  }
  let ss = 0;
  for (const v of values) ss += (v - m) ** 2;
  const half = t975(n - 1) * Math.sqrt(ss / (n - 1)) / Math.sqrt(n);
  return { mean: m, low: m - half, high: m + half };
}

/** Wilson's 95% interval of `x` successes in `n` (fractional successes allowed), exact 0 and 1 at the ends. */
export function wilson(x: number, n: number): [number, number] {
  const p = x / n;
  const z2 = Z * Z;
  const center = (p + z2 / (2 * n)) / (1 + z2 / n);
  const half = Z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n);
  return [p === 0 ? 0 : Math.max(0, center - half), p === 1 ? 1 : Math.min(1, center + half)];
}

/** What a range of a difference says. */
export type Outcome = "better" | "worse" | "same" | "not_worse" | "not_better" | "unclear";

export function outcome(low: number | null, high: number | null, margin?: number | null): Outcome {
  if (low === null || high === null) return "unclear";
  if (low > 0) return "better";
  if (high < 0) return "worse";
  if (margin) {
    if (low > -margin && high < margin) return "same";
    if (low > -margin) return "not_worse";
    if (high < margin) return "not_better";
  }
  return "unclear";
}

/** The fewest values whose removal takes their mean to 0 or across it. */
export function fragility(d: readonly number[]): number {
  if (!d.length) return 0;
  const m = mean(d);
  if (m === 0) return 0;
  const s = m > 0 ? 1 : -1;
  const order = d.map((_, i) => i).sort((i, j) => -s * d[i]! - -s * d[j]! || i - j);
  const left = order.map((i) => s * d[i]!);
  let total = sum(left);
  for (let k = 1; k <= d.length; k++) {
    total -= left[k - 1]!;
    const rest = d.length - k;
    if (rest === 0 || total / rest <= 0) return k;
  }
  return d.length;
}

/** A paired comparison (measures.md, *Is B better than A*). */
export interface Paired {
  diff: number | null;
  low: number | null;
  high: number | null;
  n: number;
  method: "mover-wilson" | "paired-t" | null;
  outcome: Outcome;
  better: number;
  worse: number;
  same: number;
  fragility: number;
}

/**
 * Is B better than A, case by case: the mean difference with its 95% range.
 * When every value is 0 or 1, the MOVER-Wilson interval with the
 * continuity-corrected correlation; otherwise the paired t interval (kept
 * within -1 and 1 when the values are bounded).
 */
export function paired(a: readonly number[], b: readonly number[], margin?: number | null, isBounded = false): Paired {
  if (a.length !== b.length) throw new RangeError(`paired needs one value of each per case: ${a.length} against ${b.length}`);
  const n = a.length;
  if (!n) return { diff: null, low: null, high: null, n: 0, method: null, outcome: "unclear", better: 0, worse: 0, same: 0, fragility: 0 };
  const d = b.map((y, i) => y - a[i]!);
  const diff = mean(d);
  let low: number | null = null, high: number | null = null, method: Paired["method"] = null;
  if (n >= 2) {
    if ([...a, ...b].every((v) => v === 0 || v === 1)) {
      method = "mover-wilson";
      let n11 = 0, n10 = 0, n01 = 0;
      for (let i = 0; i < n; i++) {
        if (a[i] === 1 && b[i] === 1) n11++;
        else if (a[i] === 1 && b[i] === 0) n10++;
        else if (a[i] === 0 && b[i] === 1) n01++;
      }
      const n00 = n - n11 - n10 - n01;
      const pa = (n11 + n10) / n, pb = (n11 + n01) / n;
      const [la, ua] = wilson(n11 + n10, n);
      const [lb, ub] = wilson(n11 + n01, n);
      let num = n11 * n00 - n10 * n01;
      if (num > 0) num = Math.max(num - n / 2, 0);
      const den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00);
      const phi = den === 0 ? 0 : num / Math.sqrt(den);
      const dl = Math.sqrt(Math.max(0, (pb - lb) ** 2 - 2 * phi * (pb - lb) * (ua - pa) + (ua - pa) ** 2));
      const du = Math.sqrt(Math.max(0, (ub - pb) ** 2 - 2 * phi * (ub - pb) * (pa - la) + (pa - la) ** 2));
      low = Math.max(-1, diff - dl);
      high = Math.min(1, diff + du);
    } else {
      method = "paired-t";
      let ss = 0;
      for (const v of d) ss += (v - diff) ** 2;
      const half = t975(n - 1) * Math.sqrt(ss / (n - 1)) / Math.sqrt(n);
      low = diff - half;
      high = diff + half;
      if (isBounded) {
        low = Math.max(-1, low);
        high = Math.min(1, high);
      }
    }
  }
  return { diff, low, high, n, method, outcome: outcome(low, high, margin), better: d.filter((v) => v > 0).length,
    worse: d.filter((v) => v < 0).length, same: d.filter((v) => v === 0).length, fragility: fragility(d) };
}

/** B's share of the picks, case by case. */
export interface WinRate { rate: number | null; low: number | null; high: number | null; n: number; outcome: Outcome; fragility: number }

/** B's share of each case's picks: the mean, Wilson's range on it, and what it says against one half. */
export function winRateOf(v: readonly number[], margin?: number | null): WinRate {
  const n = v.length;
  if (!n) return { rate: null, low: null, high: null, n: 0, outcome: "unclear", fragility: 0 };
  const [low, high] = n >= 2 ? wilson(sum(v), n) : [null, null];
  return { rate: mean(v), low, high, n, outcome: outcome(low === null ? null : low - 0.5, high === null ? null : high - 0.5, margin),
    fragility: fragility(v.map((x) => x - 0.5)) };
}

/** A judge combined with people (PPI++). */
export interface Combined { mean: number; low: number; high: number; n: number; unlabeled: number; lambda: number }

const toMap = (x: ReadonlyMap<string, number> | Readonly<Record<string, number>>): Map<string, number> =>
  x instanceof Map ? new Map(x) : new Map(Object.entries(x));

/**
 * A judge's values on every case combined with people's on a random few
 * (PPI++, measures.md): λ fitted to the data, two pseudo-observations, t with
 * n - 2 degrees of freedom. Null with fewer than three cases both judged.
 */
export function ppi(judge: ReadonlyMap<string, number> | Readonly<Record<string, number>>, people: ReadonlyMap<string, number> | Readonly<Record<string, number>>,
  isBounded = true, lo = 0, hi = 1): Combined | null {
  const J = toMap(judge), P = toMap(people);
  const labeled = sorted([...P.keys()].filter((k) => J.has(k)));
  const unlabeled = sorted([...J.keys()].filter((k) => !P.has(k)));
  const n = labeled.length, N = unlabeled.length;
  if (n < 3) return null;
  const y = labeled.map((k) => P.get(k)!);
  const f = labeled.map((k) => J.get(k)!);
  const fu = unlabeled.map((k) => J.get(k)!);
  const every = [...f, ...fu];
  const vf = every.length >= 2 ? variance(every) : 0;
  let lam = 0;
  if (N > 0 && vf !== 0) {
    const my = mean(y), mf = mean(f);
    let c = 0;
    for (let i = 0; i < n; i++) c += (y[i]! - my) * (f[i]! - mf);
    c /= n - 1;
    lam = Math.min(1, Math.max(0, c / ((1 + n / N) * vf)));
  }
  let h: number;
  if (isBounded) h = (hi - lo) / 2;
  else {
    const seen = [...y, ...every];
    h = (Math.max(...seen) - Math.min(...seen)) / 2;
  }
  const r = y.map((v, i) => v - lam * f[i]!);
  const mr = mean(r);
  const est = (N ? lam * mean(fu) : 0) + mr;
  let ss = 0;
  for (const x of r) ss += (x - mr) ** 2;
  const vr = (ss + 2 * h * h) / (n + 1);
  const se = Math.sqrt((N ? lam * lam * vf / N : 0) + vr / n);
  const q = t975(n - 2);
  let low = est - q * se, high = est + q * se;
  if (isBounded) {
    low = Math.max(lo, low);
    high = Math.min(hi, high);
  }
  return { mean: est, low, high, n, unlabeled: N, lambda: lam };
}

/** PPI++ on the differences B - A, with an outcome. */
export function ppiPaired(judgeA: ReadonlyMap<string, number> | Readonly<Record<string, number>>, judgeB: ReadonlyMap<string, number> | Readonly<Record<string, number>>,
  peopleA: ReadonlyMap<string, number> | Readonly<Record<string, number>>, peopleB: ReadonlyMap<string, number> | Readonly<Record<string, number>>,
  margin?: number | null, isBounded = true): (Omit<Combined, "mean"> & { diff: number; outcome: Outcome }) | null {
  const ja = toMap(judgeA), jb = toMap(judgeB), pa = toMap(peopleA), pb = toMap(peopleB);
  const jd = new Map([...ja].filter(([k]) => jb.has(k)).map(([k, v]) => [k, jb.get(k)! - v] as const));
  const pd = new Map([...pa].filter(([k]) => pb.has(k)).map(([k, v]) => [k, pb.get(k)! - v] as const));
  const out = ppi(jd, pd, isBounded, -1, 1);
  if (!out) return null;
  return { diff: out.mean, low: out.low, high: out.high, n: out.n, unlabeled: out.unlabeled, lambda: out.lambda, outcome: outcome(out.low, out.high, margin) };
}

function kappa(xs: readonly unknown[], ys: readonly unknown[]): number | null {
  const cats = [...new Set([...xs, ...ys].map((v) => JSON.stringify(v)))];
  const n = xs.length;
  let po = 0;
  for (let i = 0; i < n; i++) if (JSON.stringify(xs[i]) === JSON.stringify(ys[i])) po++;
  po /= n;
  let pe = 0;
  for (const c of cats) pe += (xs.filter((v) => JSON.stringify(v) === c).length / n) * (ys.filter((v) => JSON.stringify(v) === c).length / n);
  return pe === 1 ? null : (po - pe) / (1 - pe);
}

const category = (v: number) => (v > 0.5 ? "second" : v < 0.5 ? "first" : "tie");

/** Agreement of two sources (measures.md, *Can this judge stand in for people*). */
export interface Agreement { n: number; agree: number | null; low: number | null; high: number | null; kappa: number | null; mad: number | null; r: number | null }

export function agreementOf(a: ReadonlyMap<string, number> | Readonly<Record<string, number>>, b: ReadonlyMap<string, number> | Readonly<Record<string, number>>, choice = false): Agreement {
  const A = toMap(a), B = toMap(b);
  const keys = sorted([...A.keys()].filter((k) => B.has(k)));
  const n = keys.length;
  if (!n) return { n: 0, agree: null, low: null, high: null, kappa: null, mad: null, r: null };
  const xs = keys.map((k) => A.get(k)!), ys = keys.map((k) => B.get(k)!);
  if (choice || [...xs, ...ys].every((v) => v === 0 || v === 1)) {
    const cx = choice ? xs.map(category) : xs, cy = choice ? ys.map(category) : ys;
    const same = cx.filter((v, i) => v === cy[i]).length;
    const [low, high] = n >= 2 ? wilson(same, n) : [null, null];
    return { n, agree: same / n, low, high, kappa: kappa(cx, cy), mad: null, r: null };
  }
  const mx = mean(xs), my = mean(ys);
  let sx = 0, sy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += (xs[i]! - mx) ** 2;
    sy += (ys[i]! - my) ** 2;
    sxy += (xs[i]! - mx) * (ys[i]! - my);
  }
  sx = Math.sqrt(sx);
  sy = Math.sqrt(sy);
  return { n, agree: null, low: null, high: null, kappa: null, mad: mean(xs.map((x, i) => Math.abs(x - ys[i]!))), r: sx === 0 || sy === 0 ? null : sxy / (sx * sy) };
}

/** Whether a judge favours a position. */
export interface Positions {
  pairs: number; consistent: number | null; consistent_low: number | null; consistent_high: number | null;
  picks: number; first_wins: number | null; first_low: number | null; first_high: number | null;
}

export function positionsOf(ratings: Iterable<Rec>, crit: Rec, judgeId: string): Positions {
  const js = counting(ratings, crit).filter((j) => j.calls.length === 2 && j.source[0] === "judge" && j.source[1] === judgeId);
  const byPair = new Map<string, Map<string, unknown>>();
  for (const j of js) {
    const k = tuple(sorted(j.calls));
    const m = byPair.get(k) ?? new Map<string, unknown>();
    m.set(tuple(j.calls), j.value);
    byPair.set(k, m);
  }
  const both = sorted(byPair.keys()).filter((k) => byPair.get(k)!.size === 2);
  const same = both.filter((k) => new Set([...byPair.get(k)!.values()]).size === 1).length;
  const decided = js.filter((j) => j.value !== "tie");
  const first = decided.filter((j) => j.value === j.calls[0]).length;
  const [cl, ch] = both.length >= 2 ? wilson(same, both.length) : [null, null];
  const [fl, fh] = decided.length >= 2 ? wilson(first, decided.length) : [null, null];
  return { pairs: both.length, consistent: both.length ? same / both.length : null, consistent_low: cl, consistent_high: ch,
    picks: decided.length, first_wins: decided.length ? first / decided.length : null, first_low: fl, first_high: fh };
}

/** How much a setup agrees with itself. */
export interface SelfAgreement { cases: number; mean: number | null; low: number | null; high: number | null; spread: number | null }

export function selfAgreementOf(perCase: ReadonlyMap<string, readonly number[]> | Readonly<Record<string, readonly number[]>>): SelfAgreement {
  const m = perCase instanceof Map ? perCase : new Map(Object.entries(perCase));
  const cases = sorted(m.keys()).filter((k) => m.get(k)!.length >= 2);
  if (!cases.length) return { cases: 0, mean: null, low: null, high: null, spread: null };
  const agree: number[] = [], within: number[] = [];
  for (const k of cases) {
    const vs = m.get(k)!;
    const pairs: number[] = [];
    for (let i = 0; i < vs.length; i++) for (let j = i + 1; j < vs.length; j++) pairs.push(1 - Math.abs(vs[i]! - vs[j]!));
    agree.push(mean(pairs));
    const mm = mean(vs);
    let s = 0;
    for (const v of vs) s += (v - mm) ** 2;
    within.push(s / vs.length);
  }
  const iv = interval(agree);
  const clip = (x: number | null) => (x === null ? null : Math.min(1, Math.max(0, x)));
  return { cases: cases.length, mean: iv.mean, low: clip(iv.low), high: clip(iv.high), spread: Math.sqrt(mean(within)) };
}

// ------------------------------------------------------------------ Bradley–Terry

type Fit = { s: (number | null)[]; ref: number | null; comp: number[]; p: number[]; count: number[][] };

function btFit(k: number, outcomes: readonly (readonly [number, number, number])[]): Fit {
  const met = new Map<string, [number, number]>();
  for (const [i, j] of outcomes) if (i !== j) met.set(`${Math.min(i, j)},${Math.max(i, j)}`, [Math.min(i, j), Math.max(i, j)]);
  const virtual = [...met.values()].sort((x, y) => x[0] - y[0] || x[1] - y[1]).map(([i, j]) => [i, j, 0.5] as const);
  const all = [...outcomes, ...virtual];
  const adj: Set<number>[] = Array.from({ length: k }, () => new Set<number>());
  for (const [i, j] of met.values()) {
    adj[i]!.add(j);
    adj[j]!.add(i);
  }
  const ref = adj.findIndex((s) => s.size > 0);
  if (ref < 0) return { s: Array(k).fill(null), ref: null, comp: [], p: [], count: [] };
  const comp = new Set([ref]);
  const todo = [ref];
  while (todo.length) {
    const x = todo.pop()!;
    for (const y of [...adj[x]!].sort((a, b) => a - b)) {
      if (!comp.has(y)) {
        comp.add(y);
        todo.push(y);
      }
    }
  }
  const compL = [...comp].sort((a, b) => a - b);
  const wins = Array(k).fill(0) as number[];
  const count = Array.from({ length: k }, () => Array(k).fill(0) as number[]);
  for (const [i, j, s] of all) {
    if (i === j) continue;
    wins[i]! += s;
    wins[j]! += 1 - s;
    count[i]![j]! += 1;
    count[j]![i]! += 1;
  }
  let p = Array(k).fill(1) as number[];
  for (let round = 0; round < 10000; round++) {
    const next = [...p];
    for (const i of compL) {
      let denom = 0;
      for (const j of compL) if (j !== i && count[i]![j]) denom += count[i]![j]! / (p[i]! + p[j]!);
      next[i] = wins[i]! / denom;
    }
    const r = next[ref]!;
    const normed = next.map((x, idx) => (comp.has(idx) ? x / r : 1));
    let delta = 0;
    for (const i of compL) delta = Math.max(delta, Math.abs(Math.log(normed[i]!) - Math.log(p[i]!)));
    p = normed;
    if (delta < 1e-12) break;
  }
  return { s: p.map((x, i) => (comp.has(i) ? Math.log(x) : null)), ref, comp: compL, p, count };
}

function invert(m: number[][]): number[][] {
  const n = m.length;
  const a = m.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r]![c]!) > Math.abs(a[piv]![c]!)) piv = r;
    [a[c], a[piv]] = [a[piv]!, a[c]!];
    const pv = a[c]![c]!;
    a[c] = a[c]!.map((x) => x / pv);
    for (let r = 0; r < n; r++) {
      if (r !== c && a[r]![c] !== 0) {
        const f = a[r]![c]!;
        a[r] = a[r]!.map((x, i) => x - f * a[c]![i]!);
      }
    }
  }
  return a.map((row) => row.slice(n));
}

/** One side-by-side outcome: a case, two versions, and the first's share of the win. */
export type RankOutcome = readonly [string, string, string, number];

const strengths = (versions: readonly string[], outcomes: readonly RankOutcome[]): Fit => {
  const idx = new Map(versions.map((v, i) => [v, i] as const));
  return btFit(versions.length, outcomes.map(([, i, j, s]) => [idx.get(i)!, idx.get(j)!, s] as const));
};

/** A ranking (measures.md, *How do many versions rank*). */
export interface Ranking {
  versions: { version: string; strength: number | null; low: number | null; high: number | null; vs_reference: number | null }[];
  order: string[];
  reference: string | null;
  fragility: number | null;
}

/** Rank versions from side-by-side outcomes (Bradley–Terry, Hunter's MM, one virtual tie per pair that met). */
export function bradleyTerry(versions: readonly string[], outcomes: readonly RankOutcome[]): Ranking {
  const fit = strengths(versions, outcomes);
  if (fit.ref === null) {
    return { versions: versions.map((v) => ({ version: v, strength: null, low: null, high: null, vs_reference: null })), order: [], reference: null, fragility: null };
  }
  const { s, ref, comp, p, count } = fit;
  const others = comp.filter((i) => i !== ref);
  const info = others.map(() => Array(others.length).fill(0) as number[]);
  others.forEach((i, a) => {
    for (const j of comp) {
      if (j === i || !count[i]![j]) continue;
      const w = count[i]![j]! * (p[i]! / (p[i]! + p[j]!)) * (p[j]! / (p[i]! + p[j]!));
      info[a]![a]! += w;
      if (j !== ref) info[a]![others.indexOf(j)]! -= w;
    }
  });
  const inv = others.length ? invert(info) : [];
  const rows = versions.map((v, n) => {
    if (s[n] === null) return { version: v, strength: null, low: null, high: null, vs_reference: null };
    if (n === ref) return { version: v, strength: 0, low: 0, high: 0, vs_reference: 0.5 };
    const o = others.indexOf(n);
    const se = Math.sqrt(Math.max(0, inv[o]![o]!));
    return { version: v, strength: s[n]!, low: s[n]! - Z * se, high: s[n]! + Z * se, vs_reference: p[n]! / (p[n]! + 1) };
  });
  const order = versions.map((_, i) => i).filter((i) => s[i] !== null).sort((x, y) => s[y]! - s[x]! || x - y);
  return { versions: rows, order: order.map((i) => versions[i]!), reference: versions[ref]!, fragility: rankFragility(versions, outcomes, order) };
}

function rankFragility(versions: readonly string[], outcomes: readonly RankOutcome[], order: readonly number[], limit = RANK_STEPS): number | null {
  if (order.length < 2) return null;
  const leader = order[0]!;
  let kept = [...outcomes];
  for (let step = 1; step <= limit; step++) {
    const cases = sorted(new Set(kept.map((o) => o[0])));
    if (!cases.length) return null;
    let best: [number, string] | null = null;
    for (const c of cases) {
      const { s } = strengths(versions, kept.filter((o) => o[0] !== c));
      const live = versions.map((_, i) => i).filter((i) => s[i] !== null);
      if (live.length < 2 || s[leader] === null) continue;
      const top = [...live].sort((x, y) => s[y]! - s[x]! || x - y)[0]!;
      if (top !== leader) return step;
      const gap = s[leader]! - Math.max(...live.filter((i) => i !== leader).map((i) => s[i]!));
      if (gap <= 0) return step;
      if (best === null || gap < best[0] - 1e-9) best = [gap, c];
    }
    if (best === null) return null;
    const drop = best[1];
    kept = kept.filter((o) => o[0] !== drop);
  }
  return null;
}

// ------------------------------------------------------------------ scoreboard

function trees(calls: readonly Rec[], roots: readonly Rec[]): Map<string, Rec[]> {
  const best = new Map<string, Rec>();
  for (const r of calls) {
    if (r["functai_call"] !== 1 && r["functai_call"] !== 2) continue;
    const had = best.get(String(r["id"]));
    if (!had || Number(r["writer"] ?? 1) > Number(had["writer"] ?? 1)) best.set(String(r["id"]), r);
  }
  const kids = new Map<string, Rec[]>();
  for (const r of best.values()) {
    if (r["parent"]) kids.set(String(r["parent"]), [...(kids.get(String(r["parent"])) ?? []), r]);
  }
  const out = new Map<string, Rec[]>();
  for (const c of roots) {
    const tree: Rec[] = [];
    const todo = [c];
    while (todo.length) {
      const x = todo.pop()!;
      tree.push(x);
      todo.push(...(kids.get(String(x["id"])) ?? []));
    }
    out.set(String(c["id"]), tree);
  }
  return out;
}

function usageSum(ts: readonly Rec[][]): { input_tokens: number | null; output_tokens: number | null; spent_usd: number } {
  const calls = ts.flat();
  const asked = calls.filter((c) => c["model"] !== null && c["model"] !== undefined);
  const total = (key: string) => {
    const vs = asked.map((c) => ((c["usage"] ?? {}) as Rec)[key]);
    return vs.every((v) => typeof v === "number" && Number.isInteger(v)) ? sum(vs as number[]) : null;
  };
  let spent = 0;
  for (const c of calls) spent += Number(((c["spent"] ?? {}) as Rec)["usd"] ?? 0) || 0;
  return { input_tokens: total("input_tokens"), output_tokens: total("output_tokens"), spent_usd: spent };
}

function estimate(perCase: ReadonlyMap<string, number>, isBounded: boolean) {
  const vs = sorted(perCase.keys()).map((k) => perCase.get(k)!);
  const iv = interval(vs);
  let { low, high } = iv;
  if (isBounded && low !== null && high !== null) {
    low = Math.max(0, low);
    high = Math.min(1, high);
  }
  return { mean: iv.mean, low, high, cases: vs.length };
}

/** One estimate on a scoreboard. */
export interface Measure { criterion: string; aggregate: string | null; source: "people" | "people drawn" | "judge" | "combined"; judge: string | null; mean: number | null; low: number | null; high: number | null; cases: number }

/** One group (a setup, or a version) on a scoreboard. */
export interface BoardGroup { group: string | null; results: number; cases: number; failed: number; seconds: number; input_tokens: number | null; output_tokens: number | null; spent_usd: number; measures: Measure[] }

/** The contract's scoreboard (measures.md, *Scoreboard*). */
export function scoreboardOf(records: readonly Rec[], program: ProgramKey, criteria: readonly Rec[], opts: { by?: string } = {}): { groups: BoardGroup[] } {
  const by = opts.by ?? "setup";
  const [calls, ratings] = splitRecords(records);
  const res = results(calls, program, by);
  const groups: (string | null)[] = [];
  for (const r of res) if (!groups.includes(r.group)) groups.push(r.group);
  const tree = trees(calls, res.map((r) => r.call));
  const rows: BoardGroup[] = [];
  for (const g of groups) {
    const mine = res.filter((r) => r.group === g).map((r) => r.call);
    const row: BoardGroup = {
      group: g, results: mine.length, cases: new Set(res.filter((r) => r.group === g).map((r) => r.case)).size,
      failed: mine.filter((c) => c["error"] !== null && c["error"] !== undefined).length, seconds: mean(mine.map((c) => Number(c["seconds"] ?? 0))),
      ...usageSum(mine.map((c) => tree.get(String(c["id"]))!)), measures: [],
    };
    for (const crit of criteria) {
      const agg = crit["items"] !== undefined && crit["items"] !== null ? String(crit["aggregate"] ?? "mean") : null;
      const judges: [string, string | null][] = [];
      for (const j of counting(ratings, crit)) {
        if (j.source[0] === "judge" && !judges.some(([id]) => id === j.source[1])) judges.push([j.source[1], j.judge ?? null]);
      }
      const got = new Map<string, Map<string, number>>();
      const sources: [string, string | null, boolean][] = [["people", null, false], ["people", null, true], ...judges.map(([id, name]) => [id, name, false] as [string, string | null, boolean])];
      for (const [src, name, drawn] of sources) {
        const cs = valuesOf(records, program, crit, src, { by, drawn }).cases.get(g) ?? new Map<string, number>();
        got.set(tuple([src, drawn]), cs);
        if (cs.size) {
          row.measures.push({ criterion: String(crit["name"]), aggregate: agg, source: drawn ? "people drawn" : src === "people" ? "people" : "judge", judge: name,
            ...estimate(cs, bounded(crit)) });
        }
      }
      for (const [id, name] of judges) {
        const comb = ppi(got.get(tuple([id, false]))!, got.get(tuple(["people", true]))!, bounded(crit));
        if (comb) row.measures.push({ criterion: String(crit["name"]), aggregate: agg, source: "combined", judge: name, mean: comb.mean, low: comb.low, high: comb.high, cases: comb.n + comb.unlabeled });
      }
    }
    rows.push(row);
  }
  return { groups: rows };
}

/** Two groups compared on the cases both have. */
export function compareOf(records: readonly Rec[], program: ProgramKey, crit: Rec, source: string, a: string | null, b: string | null,
  opts: { by?: string; drawn?: boolean; margin?: number | null } = {}): Paired & { only_a: number; only_b: number } {
  const cs = valuesOf(records, program, crit, source, { by: opts.by, drawn: opts.drawn }).cases;
  const ca = cs.get(a) ?? new Map<string, number>(), cb = cs.get(b) ?? new Map<string, number>();
  const shared = sorted([...ca.keys()].filter((k) => cb.has(k)));
  const out = paired(shared.map((k) => ca.get(k)!), shared.map((k) => cb.get(k)!), opts.margin, bounded(crit));
  return { ...out, only_a: [...ca.keys()].filter((k) => !cb.has(k)).length, only_b: [...cb.keys()].filter((k) => !ca.has(k)).length };
}

/** A case worth a look. */
export interface Disagreement { case: string; kind: "groups" | "repeats"; spread: number; values: { group: string | null; value: number }[] }

export function disagreementsOf(records: readonly Rec[], program: ProgramKey, crit: Rec, source: string, opts: { by?: string } = {}): Disagreement[] {
  const v = valuesOf(records, program, crit, source, opts);
  const keys = sorted(new Set([...v.cases.values()].flatMap((cs) => [...cs.keys()])));
  const out: Disagreement[] = [];
  for (const k of keys) {
    const per = [...v.cases].filter(([, cs]) => cs.has(k)).map(([g, cs]) => ({ group: g, value: cs.get(k)! }));
    const vs = per.map((x) => x.value);
    if (per.length >= 2 && Math.max(...vs) - Math.min(...vs) > 0) out.push({ case: k, kind: "groups", spread: Math.max(...vs) - Math.min(...vs), values: per });
  }
  const rep = new Map<string, { group: string | null; case: string; vs: number[] }>();
  for (const r of v.results) {
    if (r.value === null) continue;
    const key = tuple([String(r.group), r.case]);
    const had = rep.get(key) ?? { group: r.group, case: r.case, vs: [] };
    had.vs.push(r.value);
    rep.set(key, had);
  }
  const g = (x: string | null) => x ?? "";
  const reps = [...rep.values()].sort((x, y) => cmp(g(x.group), g(y.group)) || cmp(x.case, y.case));
  for (const r of reps) {
    if (r.vs.length >= 2 && Math.max(...r.vs) - Math.min(...r.vs) > 0) {
      out.push({ case: r.case, kind: "repeats", spread: Math.max(...r.vs) - Math.min(...r.vs), values: r.vs.map((x) => ({ group: r.group, value: x })) });
    }
  }
  const rank = (x: Disagreement) => (x.kind === "groups" ? 0 : 1);
  return out.sort((x, y) => rank(x) - rank(y) || y.spread - x.spread || cmp(x.case, y.case)
    || cmp(x.kind === "groups" ? "" : g(x.values[0]!.group), y.kind === "groups" ? "" : g(y.values[0]!.group)));
}

// ------------------------------------------------------------------ over a log

/** Where records are: given (`records`, or a store), or a log folder (the call log by default). */
export interface RecordsOptions { folder?: string | calllog.CallStore; records?: readonly Rec[] | calllog.CallStore; since?: Date | string }

function recordsOf(opts: RecordsOptions = {}): Rec[] {
  if (opts.records !== undefined) {
    if (calllog.isStore(opts.records)) return (opts.records.read() as Rec[]).map((r) => ({ ...r }));
    return [...(opts.records as readonly Rec[])];
  }
  const [calls, ratings] = calllog.read(opts.folder ?? calllog.folderOf(effective({}).logCalls ?? true), { since: opts.since ?? null });
  return [...calls, ...ratings];
}

type ProgramLike = string | ProgramKey | { readonly name: string; readonly module?: string | undefined; readonly setup?: string };

function keyOf(p: ProgramLike): ProgramKey {
  if (typeof p === "string") return { name: p };
  const x = p as Rec;
  return { name: String(x["name"]), ...(x["module"] !== undefined ? { module: x["module"] as string | null } : {}), ...(x["file"] !== undefined && typeof x["file"] === "string" ? { file: x["file"] } : {}) };
}

/** `"people"`, or a judge's id: its setup (a program) or its code's hash (a function). */
export function sourceId(source: unknown): string {
  if (source === undefined || source === null || source === "people") return "people";
  if (typeof source === "string") return source;
  if (typeof source === "function") {
    const s = (source as unknown as Rec)["setup"];
    if (typeof s === "string" && (source as unknown as Rec)["interface"] !== undefined) return s;
    return "sha256:" + lmcc.sha256Hex(String(source));
  }
  if (typeof source === "object" && typeof (source as Rec)["setup"] === "string") return (source as Rec)["setup"] as string;
  throw new TypeError("a source is \"people\", a judge (a program or a function), or a judge's id");
}

const groupId = (g: unknown, by: string): string | null => {
  if (g === null || g === undefined) return null;
  if (typeof g === "string") return g;
  const x = g as Rec;
  return String(by === "setup" ? x["setup"] : x["version"]);
};

/** Every group of a program's results, and each criterion's estimate by each source. */
export class Scoreboard {
  readonly program: string;
  readonly by: string;
  readonly groups: BoardGroup[];
  constructor(program: string, groups: BoardGroup[], by: string) {
    this.program = program;
    this.groups = groups;
    this.by = by;
  }

  /** One row per (group, criterion, source). */
  get rows(): Rec[] {
    return this.groups.flatMap((g): Rec[] => {
      const { measures, ...base } = g;
      return measures.length ? measures.map((m) => ({ ...base, criterion: m.criterion, aggregate: m.aggregate, source: m.source, judge: m.judge,
        mean: m.mean, low: m.low, high: m.high, measured_cases: m.cases }))
        : [{ ...base, criterion: null, aggregate: null, source: null, judge: null, mean: null, low: null, high: null, measured_cases: null }];
    });
  }

  toString(): string {
    const f = (x: number | null) => (x === null ? "—" : x.toFixed(2));
    const lines = [`Scoreboard(${this.program}, ${this.groups.length} ${this.by}${this.groups.length === 1 ? "" : "s"})`];
    for (const g of this.groups) {
      const tokens = g.input_tokens === null ? "" : `, ${(g.input_tokens + (g.output_tokens ?? 0)).toLocaleString("en")} tokens`;
      lines.push(`  ${g.group ? g.group.slice(0, 19) : "(none)"}: ${g.results} results on ${g.cases} cases, ${g.failed} failed, ${g.seconds.toFixed(1)}s each${tokens}`);
      for (const m of g.measures) {
        lines.push(`      ${m.criterion}${m.aggregate ? ` [${m.aggregate}]` : ""} by ${m.source}${m.judge ? ` (${m.judge})` : ""}: ${f(m.mean)} (95% range ${f(m.low)} to ${f(m.high)}, ${m.cases} cases)`);
      }
    }
    return lines.join("\n");
  }
}

/**
 * How good every setup of a program is, by every source of judgement: for
 * each setup (or version), its results, failures, time, tokens of whole
 * trees and what code spent; then each criterion by people, people drawn (a
 * random draw's: the fair one), each judge, and each judge combined with
 * people drawn (PPI++).
 */
export function scoreboard(program: ProgramLike, criteria: readonly Criterion[] = [correct], opts: RecordsOptions & { by?: string } = {}): Scoreboard {
  const crits = criteria.map(criterionData);
  if (crits.some((c) => c["kind"] === "choice")) throw new TypeError("a choice is between two setups: use winRate or rank for it");
  const key = keyOf(program);
  return new Scoreboard(key.name, scoreboardOf(recordsOf(opts), key, crits, { by: opts.by }).groups, opts.by ?? "setup");
}

/** Is setup `b` better than `a`, on the cases both answered (by a source's ratings). */
export function compareSetups(program: ProgramLike, a: unknown, b: unknown, crit: Criterion = correct,
  opts: RecordsOptions & { source?: unknown; drawn?: boolean; margin?: number | null; by?: string } = {}) {
  const by = opts.by ?? "setup";
  return compareOf(recordsOf(opts), keyOf(program), criterionData(crit), sourceId(opts.source), groupId(a, by), groupId(b, by),
    { by, drawn: opts.drawn, margin: opts.margin });
}

/** How often setup `b` wins side by side against `a` (ties half; a failed answer loses), with its range. */
export function winRate(program: ProgramLike, a: unknown, b: unknown, crit: Criterion = preferred,
  opts: RecordsOptions & { source?: unknown; drawn?: boolean; margin?: number | null; by?: string } = {}): WinRate {
  const by = opts.by ?? "setup";
  const per = preferencesOf(recordsOf(opts), keyOf(program), criterionData(crit), sourceId(opts.source), groupId(a, by), groupId(b, by), { by, drawn: opts.drawn });
  return winRateOf([...per.values()], opts.margin);
}

/** Rank every setup of a program from side-by-side picks (Bradley–Terry), with fragility. */
export function rank(program: ProgramLike, crit: Criterion = preferred,
  opts: RecordsOptions & { source?: unknown; drawn?: boolean; by?: string; groups?: readonly unknown[] } = {}): Ranking {
  const by = opts.by ?? "setup";
  const recs = recordsOf(opts);
  const key = keyOf(program), c = criterionData(crit), src = sourceId(opts.source);
  let groups: string[];
  if (opts.groups) groups = opts.groups.map((g) => String(groupId(g, by)));
  else {
    groups = [];
    for (const r of results(splitRecords(recs)[0], key, by)) if (!groups.includes(String(r.group))) groups.push(String(r.group));
  }
  const outcomes: RankOutcome[] = [];
  groups.forEach((ga, i) => {
    for (const gb of groups.slice(i + 1)) {
      for (const [k, shareB] of preferencesOf(recs, key, c, src, ga, gb, { by, drawn: opts.drawn })) outcomes.push([k, ga, gb, 1 - shareB]);
    }
  });
  return bradleyTerry(groups, outcomes);
}

/** Can a judge stand in for people: two sources on the targets both judged. */
export function agreement(program: ProgramLike, crit: Criterion, a: unknown, b: unknown, opts: RecordsOptions & { drawn?: boolean } = {}): Agreement {
  const recs = recordsOf(opts);
  const c = criterionData(crit);
  const [calls, ratings] = splitRecords(recs);
  const mine = new Set(programCalls(calls, keyOf(program)).map((x) => String(x["id"])));
  const values = (src: string, drawn: boolean) => {
    const per = new Map<string, number[]>();
    for (const j of counting(ratings, c)) {
      if (!matches(j, src, drawn) || !j.calls.every((x) => mine.has(x))) continue;
      let k: string, x: number | null;
      if (c["kind"] === "choice") {
        const pair = sorted(j.calls);
        k = targetKey(pair, "");
        x = j.value === pair[1] ? 1 : j.value === pair[0] ? 0 : 0.5;
      } else {
        k = targetKey(j.calls, j.item);
        x = numberOf(j);
      }
      if (x !== null) per.set(k, [...(per.get(k) ?? []), x]);
    }
    return new Map([...per].map(([k, vs]) => [k, mean(vs)] as const));
  };
  const sa = sourceId(a), sb = sourceId(b);
  return agreementOf(values(sa, Boolean(opts.drawn) && sa === "people"), values(sb, Boolean(opts.drawn) && sb === "people"), c["kind"] === "choice");
}

/** Whether a judge favours a position (of the pairs it was shown both ways). */
export function positions(judge: unknown, crit: Criterion = preferred, opts: RecordsOptions = {}): Positions {
  return positionsOf(splitRecords(recordsOf(opts))[1], criterionData(crit), sourceId(judge));
}

/** How much each setup agrees with itself over cases it answered two or more times. */
export function selfAgreement(program: ProgramLike, crit: Criterion = correct, opts: RecordsOptions & { source?: unknown; by?: string } = {}): Map<string | null, SelfAgreement> {
  const v = valuesOf(recordsOf(opts), keyOf(program), criterionData(crit), sourceId(opts.source), { by: opts.by });
  const per = new Map<string | null, Map<string, number[]>>();
  for (const r of v.results) {
    if (r.value === null) continue;
    const g = per.get(r.group) ?? new Map<string, number[]>();
    g.set(r.case, [...(g.get(r.case) ?? []), r.value]);
    per.set(r.group, g);
  }
  return new Map([...per].map(([g, cs]) => [g, selfAgreementOf(cs)] as const));
}

/** Cases worth a look, or worth adding to the examples: where setups differ, then where one setup's repeats differ. */
export function disagreements(program: ProgramLike, crit: Criterion = correct, opts: RecordsOptions & { source?: unknown; by?: string } = {}): Disagreement[] {
  return disagreementsOf(recordsOf(opts), keyOf(program), criterionData(crit), sourceId(opts.source), { by: opts.by });
}

/** Every result's value under a criterion and source, and each group's value by case. */
export function measure(program: ProgramLike, crit: Criterion = correct, opts: RecordsOptions & { source?: unknown; drawn?: boolean; by?: string } = {}) {
  return valuesOf(recordsOf(opts), keyOf(program), criterionData(crit), sourceId(opts.source), { by: opts.by, drawn: opts.drawn });
}

// ------------------------------------------------------------------ judging

type ProgramJudge = Exclude<Judge, (args: JudgeArgs) => unknown>;
const isProgram = (j: unknown): j is ProgramJudge & { predict?: unknown; stream?: unknown } =>
  typeof j === "function" && typeof (j as unknown as Rec)["setup"] === "string" && (j as unknown as Rec)["interface"] !== undefined;

/** A rating's `judge`: a program by its setup (and version), code by its hash. */
export function judgeIdentity(j: unknown): Rec {
  if (isProgram(j)) return { name: j.name, setup: j.setup, version: j.version };
  if (typeof j === "function") return { name: j.name || "judge", code: "sha256:" + lmcc.sha256Hex(String(j)) };
  throw new TypeError("a judge is an AI function, a module or a function");
}

/** Ask a judge with what it may use: a program its interface's inputs, by name; code all of it. Its answer and call id. */
export async function askJudge(j: Judge, available: Rec): Promise<[unknown, string | null]> {
  if (isProgram(j)) {
    const ins: Rec = {};
    for (const f of j.interface.inputs) {
      if (Object.hasOwn(available, f.name)) ins[f.name] = available[f.name];
      else if (!f.optional) throw new TypeError(`judge ${j.name} asks for ${f.name}; a judge is given by name: ${Object.keys(available).sort().join(", ")}`);
    }
    if (typeof (j as Rec)["predict"] === "function") {
      const pred = await (j as unknown as { predict(i: Rec): Promise<Prediction<Rec, unknown>> }).predict(ins);
      return [pred.answer ?? (pred.outputs as Rec)[(j as unknown as Rec)["answerName"] as string], pred.callId];
    }
    const s = (j as unknown as { stream(i: Rec): { result: Promise<unknown>; callId: string | null } }).stream(ins);
    const value = await s.result;
    return [value, s.callId];
  }
  const mark = { case: null, repeat: null, call: null as string | null };
  const out = await calllog.MARK.run(mark, () => Promise.resolve((j as (a: JudgeArgs) => unknown)(available as JudgeArgs)));
  return [out, mark.call];
}

/** A judge's answer as a rating's value for the criterion's kind. */
export function judgedValue(crit: Rec, out: unknown): unknown {
  let v = out;
  if (v !== null && typeof v === "object" && !Array.isArray(v)) {
    const r = v as Rec;
    if (Object.hasOwn(r, "value")) v = r["value"];
    else if (Object.hasOwn(r, String(crit["name"]))) v = r[String(crit["name"])];
    else throw new TypeError(`the judge's answer for ${String(crit["name"])} is a record without "value" or "${String(crit["name"])}"`);
  }
  const k = crit["kind"];
  if (k === "verdict") {
    if (v === true || v === "right") return "right";
    if (v === false || v === "wrong") return "wrong";
    throw new TypeError(`a verdict judge answers true, false, "right" or "wrong"; not ${JSON.stringify(v)}`);
  }
  if (k === "label") {
    if (typeof v === "string" && Object.hasOwn((crit["answers"] ?? {}) as Rec, v)) return v;
    throw new TypeError(`a label judge answers one of ${Object.keys((crit["answers"] ?? {}) as Rec).join(", ")}; not ${JSON.stringify(v)}`);
  }
  if (k === "score") {
    if (typeof v !== "number") throw new TypeError(`a score judge answers a number; not ${JSON.stringify(v)}`);
    const sc = crit["scale"] as number[] | undefined;
    if (sc && !(sc[0]! <= v && v <= sc[1]!)) throw new RangeError(`a score of ${v} is outside the scale [${sc.join(", ")}]`);
    return v;
  }
  if (v === "first" || v === "second" || v === "tie") return v;
  throw new TypeError(`a choice judge answers "first", "second" or "tie"; not ${JSON.stringify(v)}`);
}

/**
 * A rating record: format 1 when format 1 can say it (a person's or an
 * account's verdict on a whole call under `correct`), else format 2.
 */
export function ratingRecord(target: string | readonly [string, string], crit: Rec, value: unknown, opts: {
  who: Rec; item?: string | null; blind?: boolean; sample?: string; note?: string; reasons?: readonly string[]; origin?: string;
  answer?: unknown; outputs?: Rec; judgeCall?: string | null; corrected?: boolean;
}): Rec {
  const pair = Array.isArray(target);
  const oneFormat = crit["name"] === "correct" && crit["kind"] === "verdict" && !opts.item && !pair && !("judge" in opts.who) && !opts.judgeCall;
  let rec: Rec;
  if (oneFormat) rec = { functai_rating: 1, id: calllog.newId(), call: target, at: calllog.iso(Date.now()), ...opts.who, verdict: value };
  else {
    rec = { functai_rating: 2, id: calllog.newId(), at: calllog.iso(Date.now()) };
    if (pair) rec["calls"] = [String((target as readonly string[])[0]), String((target as readonly string[])[1])];
    else rec["call"] = String(target);
    if (opts.item) rec["item"] = opts.item;
    rec["criterion"] = crit["name"];
    if (crit["question"]) rec["question"] = crit["question"];
    rec["kind"] = crit["kind"];
    if (crit["kind"] === "label") rec["answers"] = { ...(crit["answers"] as Rec) };
    if (crit["kind"] === "score" && crit["scale"]) rec["scale"] = [...(crit["scale"] as number[])];
    rec["value"] = value;
    Object.assign(rec, opts.who);
    if (opts.judgeCall) rec["judge_call"] = opts.judgeCall;
  }
  if (opts.corrected) {
    if ("answer" in opts) rec["answer"] = toJson(opts.answer)[0];
    if (opts.outputs && Object.keys(opts.outputs).length) rec["outputs"] = Object.fromEntries(Object.entries(opts.outputs).map(([k, x]) => [k, toJson(x)[0]]));
  }
  if (opts.reasons?.length) rec["reasons"] = [...opts.reasons];
  if (opts.note) rec["note"] = opts.note;
  if (opts.origin) rec["origin"] = opts.origin;
  if (opts.sample) rec["sample"] = opts.sample;
  if (opts.blind !== undefined) rec["blind"] = Boolean(opts.blind);
  return rec;
}

/** Keep rating records where the calls are: a store or list given as records, a folder, or the call log. */
async function keep(ratings: readonly Rec[], opts: RecordsOptions): Promise<void> {
  if (!ratings.length) return;
  if (Array.isArray(opts.records)) {
    (opts.records as Rec[]).push(...ratings.map((r) => ({ ...r })));
    return;
  }
  const target = calllog.isStore(opts.records) ? opts.records : opts.folder;
  for (const r of ratings) await calllog.writeRating(r, target, effective({}));
}

function itemsOf(outputs: unknown, items: string): unknown[] {
  const lst = pointer(outputs, items);
  return Array.isArray(lst) ? lst : [];
}

async function pool<T, R>(items: readonly T[], concurrency: number, fn: (x: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      out[i] = await fn(items[i]!);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(concurrency, items.length || 1)) }, worker));
  return out;
}

function callRecords(calls: readonly CallLike[], opts: RecordsOptions): Rec[] {
  if (calls.every((c) => typeof c === "object" && c !== null && "functai_call" in c)) return calls.map((c) => ({ ...(c as Rec) }));
  const known = new Map(recordsOf(opts).filter((r) => "functai_call" in r).map((r) => [String(r["id"]), r] as const));
  return calls.map((c) => {
    if (typeof c === "object" && c !== null && "functai_call" in c) return { ...(c as Rec) };
    const id = idOf(c);
    const r = known.get(id);
    if (!r) throw new Error(`no logged call ${id}: is the call log on (logCalls), and flushed (flushCalls)?`);
    return r;
  });
}

/**
 * Have a judge rate logged calls, and keep its ratings in the log (naming
 * it by its setup, or its code's hash). An item criterion has every item
 * judged.
 */
export async function judge(j: Judge, calls: readonly CallLike[], crit: Criterion,
  opts: RecordsOptions & { sample?: string; expected?: Readonly<Record<string, unknown>>; concurrency?: number } = {}): Promise<Rec[]> {
  const c = criterionData(crit);
  if (c["kind"] === "choice") throw new TypeError("a choice is between two answers: judgePairs");
  const recs = callRecords(calls, opts);
  const who = { judge: judgeIdentity(j) };
  const made = await pool(recs, opts.concurrency ?? 4, async (call) => {
    if ((call["error"] !== null && call["error"] !== undefined) || !call["outputs"]) return [];
    const outputs = call["outputs"] as Rec;
    const answerName = (((call["program"] ?? {}) as Rec)["answer"] as string) || "result";
    const inputs = (call["inputs"] ?? {}) as Rec;
    const base: Rec = { ...inputs, inputs, outputs, answer: outputs[answerName], expected: opts.expected?.[String(call["id"])] };
    const targets: [string | null, unknown][] = c["items"] === undefined ? [[null, undefined]] : itemsOf(outputs, String(c["items"])).map((it, i) => [`${String(c["items"])}/${i}`, it]);
    const out: Rec[] = [];
    for (const [item, it] of targets) {
      const [got, jc] = await askJudge(j, item === null ? base : { ...base, item: it });
      out.push(ratingRecord(String(call["id"]), c, judgedValue(c, got), { who, item, sample: opts.sample, origin: "judge", judgeCall: jc }));
    }
    return out;
  });
  const written = made.flat();
  await keep(written, opts);
  return written;
}

/**
 * Have a judge pick the better of two answers to the same case, in both
 * orders (judges favour a position; `positions` measures it). It is given
 * `first` and `second` (the answers), the case's inputs, `first_outputs`
 * and `second_outputs`, and answers "first", "second" or "tie".
 */
export async function judgePairs(j: Judge, pairs: readonly (readonly [CallLike, CallLike])[], crit: Criterion = preferred,
  opts: RecordsOptions & { bothOrders?: boolean; sample?: string; concurrency?: number } = {}): Promise<Rec[]> {
  const c = criterionData(crit);
  if (c["kind"] !== "choice") throw new TypeError(`judgePairs judges a choice; ${String(c["name"])} is a ${String(c["kind"])}`);
  const flat = callRecords(pairs.flat(), opts);
  const byId = new Map(flat.map((r) => [String(r["id"]), r] as const));
  const ids = pairs.map(([a, b]) => [idOf(a), idOf(b)] as const);
  const jobs = [...ids, ...((opts.bothOrders ?? true) ? ids.map(([a, b]) => [b, a] as const) : [])];
  const who = { judge: judgeIdentity(j) };
  const made = await pool(jobs, opts.concurrency ?? 4, async ([x, y]) => {
    const first = byId.get(x)!, second = byId.get(y)!;
    if ((first["error"] ?? null) !== null || (second["error"] ?? null) !== null) return null;
    const name = (((first["program"] ?? {}) as Rec)["answer"] as string) || "result";
    const inputs = (first["inputs"] ?? {}) as Rec;
    const avail: Rec = { ...inputs, inputs, first: ((first["outputs"] ?? {}) as Rec)[name], second: ((second["outputs"] ?? {}) as Rec)[name],
      first_outputs: first["outputs"], second_outputs: second["outputs"] };
    const [got, jc] = await askJudge(j, avail);
    const v = judgedValue(c, got);
    return ratingRecord([x, y], c, v === "first" ? x : v === "second" ? y : "tie", { who, sample: opts.sample, origin: "judge", judgeCall: jc });
  });
  const written = made.filter((r): r is Rec => r !== null);
  await keep(written, opts);
  return written;
}

function personOf(by: string | undefined): Rec {
  const person = by ?? (calllog.callerOf(effective({}))["user"] as string | undefined);
  return person ? { by: person } : { account: (calllog.processJson()["user"] as string | null) ?? "unknown" };
}

/** A person's judgement under a criterion (format 2 unless format 1 can say it), kept in the log. */
export function rateItem(call: CallLike, value: unknown, opts: {
  criterion: Criterion; item?: number | string; by?: string; blind?: boolean; sample?: string; note?: string; reasons?: string[];
  origin?: string; folder?: string | calllog.CallStore; answer?: unknown; outputs?: Rec;
}): Rec {
  const c = criterionData(opts.criterion);
  if (c["kind"] === "choice") throw new TypeError("a choice is between two answers: prefer");
  let item: string | null = null;
  if (typeof opts.item === "number") {
    if (c["items"] === undefined) throw new TypeError(`${String(c["name"])} judges whole answers; item ${opts.item} of what?`);
    item = `${String(c["items"])}/${opts.item}`;
  } else if (opts.item !== undefined) {
    if (!opts.item.startsWith("/")) throw new TypeError("item is the item's index in the criterion's list, or a JSON Pointer (\"/insights/3\")");
    item = opts.item;
  }
  if (item === null && c["items"] !== undefined) throw new TypeError(`${String(c["name"])} judges each item of ${String(c["items"])}: say which (item: 3)`);
  const v = value === null ? null : judgedValue(c, value);
  const corrected = "answer" in opts || Boolean(opts.outputs && Object.keys(opts.outputs).length);
  if (corrected && v !== "wrong") throw new TypeError("a correction goes with a verdict of \"wrong\"");
  const rec = ratingRecord(idOf(call), c, v, { who: personOf(opts.by), item, blind: opts.blind, sample: opts.sample, note: opts.note,
    reasons: opts.reasons, origin: opts.origin, ...("answer" in opts ? { answer: opts.answer } : {}), outputs: opts.outputs, corrected });
  calllog.keepRating(rec, opts.folder, effective({}));
  return rec;
}

/**
 * Say which of two answers to the same case is better: `a` and `b` in the
 * order they were shown, the `winner` one of them, "tie", or null
 * (withdraw). `blind`: you were not told which version made which (show them
 * in `shownOrder`).
 */
export function prefer(a: CallLike, b: CallLike, winner: CallLike | "tie" | null,
  opts: { criterion?: Criterion; by?: string; blind?: boolean; sample?: string; note?: string; folder?: string | calllog.CallStore } = {}): Rec {
  const c = criterionData(opts.criterion ?? preferred);
  if (c["kind"] !== "choice") throw new TypeError(`${String(c["name"])} is not a choice`);
  const ia = idOf(a), ib = idOf(b);
  if (ia === ib) throw new TypeError("two answers are compared, not one with itself");
  let value: string | null;
  if (winner === null || winner === "tie") value = winner;
  else {
    value = idOf(winner);
    if (value !== ia && value !== ib) throw new TypeError("the winner is one of the two answers, \"tie\", or null to withdraw");
  }
  const rec = ratingRecord([ia, ib], c, value, { who: personOf(opts.by), blind: opts.blind, sample: opts.sample, note: opts.note });
  calllog.keepRating(rec, opts.folder, effective({}));
  return rec;
}

/** Two answers in the order to show them blind: by the hash of the seed and the pair, the same everywhere. */
export function shownOrder(a: CallLike, b: CallLike, seed = ""): [string, string] {
  const [x, y] = sorted([idOf(a), idOf(b)]) as [string, string];
  const flip = parseInt(lmcc.sha256Hex(`functai-shown:${seed}:${x}:${y}`).slice(0, 2), 16) % 2 === 1;
  return flip ? [y, x] : [x, y];
}

/**
 * A logged call as a row to add to a program's examples: its inputs, its
 * answer when it was right (or the correction of a "wrong" rating), and its
 * `split`, held out by the rule every language shares.
 */
export function asCase(call: CallLike, opts: RecordsOptions & { rating?: Rec; share?: number; salt?: string } = {}): Rec {
  const rec = callRecords([call], opts)[0]!;
  if (rec["content"] !== true || ((rec["saw"] ?? []) as unknown[]).length) throw new Error("only a call whose inputs were all recorded, and that saw no earlier calls, makes a case");
  const inputs = (rec["inputs"] ?? {}) as Rec;
  const row: Rec = { ...inputs };
  const answer = (((rec["program"] ?? {}) as Rec)["answer"] as string) || "result";
  let r = opts.rating;
  if (!r) {
    const mine = recordsOf(opts).filter((x) => "functai_rating" in x && x["call"] === rec["id"] && !("judge" in x) && judgement(x)?.criterion === "correct");
    r = mine.sort((x, y) => cmp(String(x["at"]), String(y["at"])) || cmp(String(x["id"]), String(y["id"]))).at(-1);
  }
  if (r) {
    const v = r["verdict"] ?? r["value"];
    const outputs = (rec["outputs"] ?? null) as Rec | null;
    if (v === "right" && outputs && Object.hasOwn(outputs, answer)) row[answer] = outputs[answer];
    else if (v === "wrong" && "answer" in r) row[answer] = r["answer"];
    for (const [k, x] of Object.entries((r["outputs"] ?? {}) as Rec)) if (!Object.hasOwn(row, k)) row[k] = x;
  }
  const names = Object.keys(inputs);
  if (names.includes("split")) throw new Error("the program has an input named split: its rows cannot also say their side");
  row["split"] = splitOf(row, names, { share: opts.share, salt: opts.salt });
  return row;
}
