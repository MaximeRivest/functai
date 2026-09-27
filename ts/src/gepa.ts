/**
 * GEPA: the instruction rewritten from the function's mistakes (Agrawal et
 * al., 2025), with functai's changes. The algorithm, what it changes and why
 * are in design/04-gepa.md; Python's `GEPA` and R's `gepa()` are the same
 * algorithm, and its two prompts send the same words.
 */

import { evaluate, exactMatch, type Metric } from "./evaluate.ts";
import { ai, type AnyAIFunction as AIFunction, type Expected, type Row } from "./fn.ts";
import { random } from "./optimize.ts";
import { newId } from "./calllog.ts";
import { withSettings, type Settings } from "./settings.ts";
import { t } from "./shapes.ts";
import { normalize, trimWhite } from "./text.ts";

type Rec = Record<string, unknown>;

const REFLECT_TEXT = `You improve the instruction of a function that a language model runs. You are given what the
function takes and returns, its current instruction, and cases it was run on: each with its inputs,
the answer it gave, its score and feedback. Find what the instruction is missing, or gets wrong, that
explains the mistakes, and write an improved instruction. Write general rules a careful person could
follow on new cases; never copy an input or describe these particular cases. Keep what already works.
The instruction is everything the model is told besides the inputs: keep the task, and say what each
output must be. Instructions listed as tried did not do better: try something different. Reply with
the new instruction only.`;

const COMBINE_TEXT = `Two instructions for the same function each get right some cases the other gets wrong. Write one
instruction that keeps what makes each of them right, without repeating itself. The instruction is
everything the model is told besides the inputs: keep the task, and say what each output must be.
Reply with the new instruction only.`;

/** Words about one answer: `(row, prediction, error) => text`; `prediction` is null when the call failed. */
export type Feedback<R = Rec> = (row: R, prediction: Rec | null, error: string | null) => string;

export interface GepaOptions<F = AIFunction, R = Rec> {
  /** Where the right answers are, as in `evaluate`: a column for the answer, or `{output: column}`. */
  expected?: Expected<F, R>;
  /** A score per row, 1 meaning right (default: exact_match against the right answers). */
  metric?: Metric<R>;
  /** Words per row (default: "right", "wrong: the right answer is …", or the call's error). */
  feedback?: Feedback<R>;
  /** Calls of the function at most (default 300). */
  budget?: number;
  /** Feedback rows per new instruction (default 4). */
  minibatch?: number;
  /** The model that writes instructions (`"gpt-6-sol"`); default the function's own. */
  teacher?: string;
  /** Rows to choose with, instead of half of the rows. */
  selection?: readonly R[];
  seed?: number;
  /** Calls in flight at once (default 8). */
  concurrency?: number;
}

/** One instruction the search tried. */
export interface Trial {
  /** Its number in the pool (0: the written one), or null when it did not join. */
  candidate: number | null;
  kind: "written" | "reflect" | "combine";
  parents: number[];
  /** How many of the minibatch rows its parent (the better of two, for a combine) and it got right. */
  minibatchParent: number | null;
  minibatch: number | null;
  /** Its mean on the choosing rows (optimistic for the one chosen: it was chosen on them). */
  score: number | null;
  length: number;
  note: string;
  /** Calls of the function so far. */
  calls: number;
  instruction: string;
  chosen: boolean;
}

/** What `gepa` returns: the improved copy, and the search that made it. */
export interface GepaResult<F> {
  /** A copy with the best instruction (the function's own when nothing beat it). */
  fn: F;
  /** Every instruction tried. */
  trials: Trial[];
  /** Calls of the function. */
  calls: number;
  /** Instructions the teacher wrote. */
  reflections: number;
}

interface Search {
  trials: Trial[];
  calls: number;
  reflections: number;
}

interface Candidate {
  instruction: string;
  parents: number[];
  kind: Trial["kind"];
  scores: number[];
  tried: string[];
}

interface Result {
  score: number;
  pred: Rec | null;
  error: string | null;
}

const mean = (xs: readonly number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

/**
 * Rewrite the instruction from the function's mistakes. A teacher model reads
 * the function's answers on a few rows with feedback in words ("wrong: the
 * right answer is billing") and writes a better instruction; the best of the
 * instructions it writes is kept, chosen on rows it is never shown (half the
 * rows, or `selection`). Candidates that are best on at least one choosing
 * row stay in the pool, so ideas that fix different mistakes both survive,
 * and every fourth step two of them are combined. The teacher sees what it
 * tried that failed; a proposal that copies an input is dropped; ties go to
 * the shorter instruction; no row runs twice for one instruction.
 *
 * Returns `{ fn, trials }`: an improved copy (with the written instruction
 * when nothing beat it) and the search. Scores on the choosing rows flatter the
 * one chosen: measure it with `evaluate` on rows it never saw.
 */
export async function gepa<F extends AIFunction, R extends Row<F>>(fn: F, rows: readonly R[], opts: GepaOptions<F, R> = {}): Promise<GepaResult<F>> {
  const budget = opts.budget ?? 300;
  const minibatch = Math.max(1, opts.minibatch ?? 4);
  const seed = opts.seed ?? 0;
  const rng = random(seed);
  const outputs = fn.definition.outputs.map((f) => f.name);
  const inputs = fn.definition.inputs.map((f) => f.name);
  const mapping: Record<string, string> = typeof opts.expected === "string" ? { [fn.answerName]: opts.expected }
    : (opts.expected as Record<string, string> | undefined) ?? Object.fromEntries(outputs.filter((n) => rows.some((r) => n in r)).map((n) => [n, n]));
  if (!opts.metric && !Object.keys(mapping).length) {
    throw new Error(`gepa needs the right answers: a column named like an output (${outputs.join(", ")}), or expected`);
  }
  let selectRows: readonly R[];
  let feedRows: readonly R[];
  if (opts.selection) {
    selectRows = opts.selection;
    feedRows = rows;
  } else {
    if (rows.length < 2) throw new Error("gepa needs at least 2 rows: some to learn from, some to choose with");
    const shuffled = shuffle([...rows], rng);
    const half = Math.floor(shuffled.length / 2);
    selectRows = shuffled.slice(0, half);
    feedRows = shuffled.slice(half);
  }
  const feedback = (opts.feedback ?? defaultFeedback(mapping, outputs.length === 1)) as Feedback;
  const run = newId();
  const written = fn.instructions;
  const fields = fieldsText(fn);
  const own = fn.settings;
  const metaSettings: Settings = {
    ...(own.router ? { router: own.router } : {}),
    ...(own.logCalls !== undefined ? { logCalls: own.logCalls } : {}),
    ...(own.logContent !== undefined ? { logContent: own.logContent } : {}),
    ...((opts.teacher ?? own.lm) ? { lm: opts.teacher ?? own.lm } : {}),
  };
  const meta = { definedIn: "functai.meta", includeFnName: false, adapter: "xml", module: "predict" as const, temperature: 1, ...metaSettings };
  const tagged = <R>(f: () => R): R => withSettings({ caller: { optimization: run } }, f);
  const reflect = ai("_reflect", { description: REFLECT_TEXT, input: { fields: t.string(), instruction: t.string(), cases: t.string(), tried: t.string() }, output: t.string(), ...meta });
  const combine = ai("_combine", { description: COMBINE_TEXT, input: { fields: t.string(), first: t.string(), second: t.string() }, output: t.string(), ...meta });

  const search: Search = { trials: [], calls: 0, reflections: 0 };
  const memo = new Map<string, Result>();
  const key = (set: string, i: number, text: string) => `${set}\r${i}\r${text}`;

  /** Score, prediction and error of each row for an instruction; each (instruction, row) runs once. */
  const scoresOf = async (text: string, set: "feed" | "select", idx: readonly number[]): Promise<Result[]> => {
    const source = set === "feed" ? feedRows : selectRows;
    const todo = idx.filter((i) => !memo.has(key(set, i, text)));
    if (todo.length) {
      const candidate = text === written ? fn : withInstruction(fn, text);
      const ev = await tagged(() => evaluate(candidate, todo.map((i) => source[i]!) as Row<F>[], {
        expected: opts.expected, metric: opts.metric, concurrency: opts.concurrency,
      }));
      search.calls += todo.length;
      const scores = ev.scores();
      todo.forEach((i, j) => {
        const r = ev.rows[j]!;
        memo.set(key(set, i, text), { score: scores[j]!, pred: r.outputs, error: r.error });
      });
    }
    return idx.map((i) => memo.get(key(set, i, text))!);
  };
  const batchSum = async (text: string, batch: readonly number[]) => (await scoresOf(text, "feed", batch)).reduce((a, r) => a + r.score, 0);
  const allSelect = selectRows.map((_, i) => i);
  const scoreSelect = async (c: Candidate) => {
    c.scores = (await scoresOf(c.instruction, "select", allSelect)).map((r) => r.score);
  };
  const record = (c: Candidate, k: number | null, before: number | null, after: number | null, note: string) => {
    search.trials.push({
      candidate: k, kind: c.kind, parents: c.parents, minibatchParent: before, minibatch: after,
      score: k === null ? null : mean(c.scores), length: c.instruction.length, note, calls: search.calls,
      instruction: c.instruction, chosen: false,
    });
  };
  let order: number[] = [];
  const nextBatch = (): number[] => {
    const batch: number[] = [];
    while (batch.length < Math.min(minibatch, feedRows.length)) {
      if (!order.length) order = shuffle(feedRows.map((_, i) => i), rng);
      const i = order.pop()!;
      if (!batch.includes(i)) batch.push(i);
    }
    return batch;
  };
  const candidate = (text: string, parents: number[], kind: Candidate["kind"]): Candidate => ({ instruction: text, parents, kind, scores: [], tried: [] });

  const pool: Candidate[] = [candidate(written, [], "written")];
  await scoreSelect(pool[0]!);
  record(pool[0]!, 0, null, null, "the written instruction");
  const admit = async (child: Candidate, before: number, after: number) => {
    if (search.calls + selectRows.length > budget) {
      record(child, null, before, after, "better on the minibatch; no budget left to score it");
      return;
    }
    await scoreSelect(child);
    pool.push(child);
    record(child, pool.length - 1, before, after, "joined the pool");
  };
  const isNew = (text: string) => Boolean(text) && !pool.some((c) => c.instruction === text);
  const solved = (c: Candidate) => feedRows.every((_, i) => (memo.get(key("feed", i, c.instruction))?.score ?? 0) >= 1);

  let step = 0;
  while (search.calls + 2 * minibatch <= budget && step < budget) {     // steps: a bound when rows are cached
    step++;
    const front = frontier(pool.map((c) => c.scores));
    if ([...front.keys()].every((k) => solved(pool[k]!))) {
      record(pool[0]!, null, null, null, "right on every feedback row: no mistake left to learn from");
      break;
    }
    const pair = step % 4 === 0 ? bestPair(pool.map((c) => c.scores), front) : null;
    if (pair) {
      const [a, b] = pair;
      const text = trimWhite(String(await tagged(() => combine({ fields, first: pool[a]!.instruction, second: pool[b]!.instruction })) ?? ""));
      search.reflections++;
      const child = candidate(text, [a, b], "combine");
      if (!isNew(text)) { record(child, null, null, null, "no new instruction"); continue; }
      const batch = nextBatch();
      const before = Math.max(await batchSum(pool[a]!.instruction, batch), await batchSum(pool[b]!.instruction, batch));
      const after = await batchSum(text, batch);
      if (after >= before) await admit(child, before, after);
      else record(child, null, before, after, "worse on the minibatch than its better parent");
      continue;
    }
    const k = pick(front, rng);
    const parent = pool[k]!;
    const batch = nextBatch();
    const results = await scoresOf(parent.instruction, "feed", batch);
    const before = results.reduce((s, r) => s + r.score, 0);
    if (results.every((r) => r.score >= 1)) continue;                  // nothing to learn from these rows
    const tried = parent.tried.slice(-3);
    const text = trimWhite(String(await tagged(() => reflect({
      fields, instruction: parent.instruction, cases: casesText(feedRows, batch, results, inputs, outputs, feedback),
      tried: tried.length ? tried.map((x, i) => `Tried ${i + 1}:\n${x}`).join("\n\n") : "(none)",
    })) ?? ""));
    search.reflections++;
    const child = candidate(text, [k], "reflect");
    if (!isNew(text)) { record(child, null, before, null, "no new instruction"); continue; }
    if (copiesAnInput(text, feedRows, inputs)) {
      parent.tried.push(`${text}\n(dropped: it copied an input instead of stating a rule)`);
      record(child, null, before, null, "copied an input: dropped");
      continue;
    }
    const after = await batchSum(text, batch);
    if (after > before) await admit(child, before, after);
    else {
      parent.tried.push(text);
      record(child, null, before, after, "not better on the minibatch");
    }
  }
  let best = 0;                                                        // ties go to the shorter instruction
  pool.forEach((c, i) => {
    const b = pool[best]!;
    if (mean(c.scores) > mean(b.scores) || (mean(c.scores) === mean(b.scores) && c.instruction.length < b.instruction.length)) best = i;
  });
  for (const tr of search.trials) tr.chosen = tr.candidate === best;
  const out = (best === 0 ? fn.using({}) : withInstruction(fn, pool[best]!.instruction)) as F;
  return { fn: out, ...search };
}

function withInstruction<F extends AIFunction>(fn: F, text: string): F {
  const copy = fn.using({}) as F;
  copy.instructions = text;
  return copy;
}

function shuffle<T>(items: T[], rng: () => number): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j]!, items[i]!];
  }
  return items;
}

/** A frontier candidate, with probability proportional to the rows it is best on. */
function pick(front: Map<number, number>, rng: () => number): number {
  const total = [...front.values()].reduce((a, b) => a + b, 0);
  let x = rng() * total;
  for (const [k, w] of front) {
    x -= w;
    if (x < 0) return k;
  }
  return [...front.keys()].pop()!;
}

// ------------------------------------------------------------------ parts (the same in Python and R)

/** The candidates on the Pareto frontier (best on at least one row, dominated by none), with how many rows each is best on. */
export function frontier(scores: readonly (readonly number[])[]): Map<number, number> {
  const n = scores[0]?.length ?? 0;
  const wins = new Map<number, number>();
  for (let r = 0; r < n; r++) {
    const best = Math.max(...scores.map((s) => s[r]!));
    scores.forEach((s, k) => { if (s[r] === best) wins.set(k, (wins.get(k) ?? 0) + 1); });
  }
  if (n === 0) wins.set(0, 1);
  const dominated = (a: number) => [...wins.keys()].some((b) => b !== a
    && scores[b]!.every((v, r) => v >= scores[a]![r]!) && scores[b]!.some((v, r) => v > scores[a]![r]!));
  return new Map([...wins].filter(([k]) => !dominated(k)));
}

/** Two frontier candidates that each win rows the other loses: the pair with the most such rows on its weaker side. */
export function bestPair(scores: readonly (readonly number[])[], front: Map<number, number>): [number, number] | null {
  const ks = [...front.keys()].sort((a, b) => a - b);
  let best = 0;
  let pair: [number, number] | null = null;
  for (let x = 0; x < ks.length; x++) {
    for (let y = x + 1; y < ks.length; y++) {
      const a = scores[ks[x]!]!, b = scores[ks[y]!]!;
      const w = Math.min(a.filter((v, r) => v > b[r]!).length, b.filter((v, r) => v > a[r]!).length);
      if (w > best) { best = w; pair = [ks[x]!, ks[y]!]; }
    }
  }
  return pair;
}

function answerText(v: unknown): string {
  return typeof v === "string" ? v : JSON.stringify(v);
}

function defaultFeedback(mapping: Record<string, string>, single: boolean): Feedback {
  return (row, pred, error) => {
    if (error !== null || pred === null) return `the call failed: ${error}`;
    const wrong = Object.keys(mapping).filter((k) => exactMatch({ [k]: row[mapping[k]!] }, { [k]: pred[k] })["exact_match"] !== 1);
    if (!wrong.length) return "right";
    return "wrong: " + wrong.map((k) => `the right ${single || k === "result" ? "answer" : k} is ${answerText(row[mapping[k]!])}`).join("; ");
  };
}

function casesText(rows: readonly Rec[], batch: readonly number[], results: readonly Result[], inputs: readonly string[],
                   outputs: readonly string[], feedback: Feedback): string {
  const lines: string[] = [];
  batch.forEach((i, n) => {
    const row = rows[i]!;
    const r = results[n]!;
    lines.push(`Case ${n + 1}`);
    for (const k of inputs) if (k in row) lines.push(`  ${k}: ${answerText(row[k])}`);
    if (r.pred === null) lines.push("  answer given: (none)");
    else for (const k of outputs) lines.push(`  answer given${k === "result" ? "" : ` ${k}`}: ${answerText(r.pred[k])}`);
    lines.push(`  score: ${r.score}`, `  feedback: ${feedback(row, r.pred, r.error)}`);
  });
  return lines.join("\n");
}

/** A JSON Schema shape in words, for the teacher. */
export function shapeWords(shape: Rec): string {
  const options = (shape["anyOf"] ?? shape["oneOf"]) as Rec[] | undefined;
  if (options) {
    const kept = options.filter((o) => o["type"] !== "null");
    const words = kept.map(shapeWords).join(" or ") || "nothing";
    return words + (kept.length < options.length ? ", or nothing" : "");
  }
  if (Array.isArray(shape["enum"])) return "one of " + (shape["enum"] as unknown[]).map(answerText).join(", ");
  switch (shape["type"]) {
    case "array": return "a list of " + shapeWords((shape["items"] as Rec) ?? {});
    case "object": {
      const props = (shape["properties"] ?? {}) as Record<string, Rec>;
      const names = Object.keys(props);
      return names.length ? "a record of " + names.map((k) => `${k} (${shapeWords(props[k]!)})`).join(", ") : "an object";
    }
    case "string": return "text";
    case "integer": return "a whole number";
    case "number": return "a number";
    case "boolean": return "true or false";
    default: return "a value";
  }
}

/** What a function takes and returns, one field a line, with its type and words. */
export function fieldsText(fn: AIFunction): string {
  const line = (f: { name: string; shape: Rec; desc?: string | null }) => `- ${f.name}: ${shapeWords(f.shape)}${f.desc ? `. ${f.desc}` : ""}`;
  return ["Inputs:", ...fn.definition.inputs.map(line), "Outputs:", ...fn.definition.outputs.map(line)].join("\n");
}

/** Does the instruction quote an input of 30 characters or more, verbatim (case and spacing ignored)? */
export function copiesAnInput(text: string, rows: readonly Rec[], inputs: readonly string[], atLeast = 30): boolean {
  const said = normalize(text);
  return rows.some((row) => inputs.some((k) => {
    const v = row[k];
    if (typeof v !== "string") return false;
    const n = normalize(v);
    return n.length >= atLeast && said.includes(n);
  }));
}
