/**
 * The arithmetic of training on a service (contract/training.md §6 to §10):
 * training examples ("datums") for each method, advantages, the energy
 * score, losses written as per-token weights, and choosing the checkpoint
 * and temperature to keep. Pure functions; every language computes the same
 * numbers (contract/cases/training/).
 */

import { type Chunk, chunksLength } from "./students.ts";

/** What a service trains on: the input chunks and one list per loss input (rows of K for several targets). */
export interface Datum {
  readonly modelInput: Chunk[];
  readonly lossFnInputs: Record<string, number[] | number[][]>;
}

/** Positions of chunks: token ids, null where a picture is. */
export function positions(chunks: readonly Chunk[]): (number | null)[] {
  const out: (number | null)[] = [];
  for (const c of chunks) {
    if ("text" in c) out.push(...c.text);
    else for (let i = 0; i < c.tokens; i++) out.push(null);
  }
  return out;
}

/** Adjacent text chunks joined, empty ones dropped. */
export function merged(chunks: readonly Chunk[]): Chunk[] {
  const out: Chunk[] = [];
  for (const c of chunks) {
    if ("text" in c) {
      if (!c.text.length) continue;
      const last = out.at(-1);
      if (last && "text" in last) out[out.length - 1] = { text: [...last.text, ...c.text] };
      else out.push({ text: [...c.text] });
    } else {
      out.push(c);
    }
  }
  return out;
}

/** The input (the chunks without their last token; trailing pictures dropped first) and the targets (tokens from the second on, 0 at pictures). */
export function shift(chunks: readonly Chunk[]): [Chunk[], number[]] {
  const cs = merged(chunks);
  while (cs.length && !("text" in cs.at(-1)!)) cs.pop();
  if (!cs.length || chunksLength(cs) < 2) throw new RangeError("a training example needs at least two tokens, the last of them text");
  const last = cs.at(-1) as { text: number[] };
  const input = [...cs.slice(0, -1), ...(last.text.length > 1 ? [{ text: last.text.slice(0, -1) }] : [])];
  const targets = positions(cs).slice(1).map((t) => (t === null ? 0 : t));
  return [input, targets];
}

/** A supervised example: the weight on every text position from the reply on. */
export function sftDatum(chunks: readonly Chunk[], replyStart: number, weight = 1): Datum {
  const cs = merged(chunks);
  const [input, targets] = shift(cs);
  const pos = positions(cs);
  const weights = targets.map((_, i) => (i + 1 >= replyStart && pos[i + 1] !== null ? weight : 0));
  return { modelInput: input, lossFnInputs: { target_tokens: targets, weights } };
}

/**
 * A sampled answer's example: the prompt's chunks then the sampled tokens;
 * the sampler's log-probabilities and the advantage (one for the answer, or
 * one per sampled token) on the sampled positions only.
 */
export function rlDatum(prompt: readonly Chunk[], sampled: readonly number[], logprobs: readonly number[], advantage: number | readonly number[]): Datum {
  const cs = merged([...prompt, { text: [...sampled] }]);
  const [input, targets] = shift(cs);
  const start = chunksLength(prompt);
  const at = (i: number) => i + 1 - start;
  const lp = targets.map((_, i) => (i + 1 >= start ? logprobs[at(i)]! : 0));
  const adv = targets.map((_, i) => (i + 1 >= start ? (typeof advantage === "number" ? advantage : advantage[at(i)]!) : 0));
  return { modelInput: input, lossFnInputs: { target_tokens: targets, logprobs: lp, advantages: adv } };
}

/**
 * An example with a weight per position of the reply (a loss written as
 * weights: DPO), `weights[k]` for the k-th position from the reply's start.
 */
export function weightedDatum(chunks: readonly Chunk[], replyStart: number, weights: (k: number) => number): Datum {
  const cs = merged(chunks);
  const [input, targets] = shift(cs);
  const pos = positions(cs);
  const w = targets.map((_, i) => (i + 1 >= replyStart && pos[i + 1] !== null ? weights(i + 1 - replyStart) : 0));
  return { modelInput: input, lossFnInputs: { target_tokens: targets, weights: w } };
}

/**
 * A decision's example: `chunks` end with the token at the choice's position
 * (any choice); row `position - 1` holds the K choices' tokens and their
 * weights, every other row 0.
 */
export function decisionDatum(chunks: readonly Chunk[], choiceTokens: readonly number[], weights: readonly number[]): Datum {
  const cs = merged(chunks);
  const [input] = shift(cs);
  const n = chunksLength(cs) - 1;
  const k = choiceTokens.length;
  const targets = Array.from({ length: n }, (_, i) => (i === n - 1 ? [...choiceTokens] : new Array<number>(k).fill(0)));
  const w = Array.from({ length: n }, (_, i) => (i === n - 1 ? [...weights] : new Array<number>(k).fill(0)));
  return { modelInput: input, lossFnInputs: { target_tokens: targets, weights: w } };
}

// ------------------------------------------------------------------ groups

/** Each reward minus its group's mean (over its population standard deviation + 1e-6 with `normalize`); null for a group that teaches nothing. */
export function advantages(rewards: readonly number[], normalize = false): number[] | null {
  if (Math.max(...rewards) - Math.min(...rewards) < 1e-9) return null;
  const mu = rewards.reduce((s, r) => s + r, 0) / rewards.length;
  let a = rewards.map((r) => r - mu);
  if (normalize) {
    const sd = Math.sqrt(rewards.reduce((s, r) => s + (r - mu) ** 2, 0) / rewards.length);
    a = a.map((x) => x / (sd + 1e-6));
  }
  return a;
}

/** KL to the starting weights: each token's advantage plus β·(mean δ − δ), the mean over every sampled token of the step. */
export function klAdjusted(adv: readonly (readonly number[])[], deltas: readonly (readonly number[])[], beta: number): number[][] {
  const all = deltas.flat();
  const mean = all.reduce((s, d) => s + d, 0) / all.length;
  return adv.map((row, i) => row.map((a, j) => a + beta * (mean - deltas[i]![j]!)));
}

/**
 * The energy score of a group of answers against references (contract/training.md §8):
 * `readable[i]`, `dRef[i][k]` = d(answer i, reference k), `dPair[i][j]` = d(answer i, answer j).
 */
export function energy(readable: readonly boolean[], dRef: readonly (readonly number[])[], dPair: readonly (readonly number[])[]): { rewards: number[]; score: number } {
  const idx = readable.flatMap((ok, i) => (ok ? [i] : []));
  const rewards = readable.map(() => -1);
  if (!idx.length) return { rewards, score: 1 };
  const mean = (xs: readonly number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
  const a = new Map(idx.map((i) => [i, mean(dRef[i]!)]));
  if (idx.length === 1) {
    rewards[idx[0]!] = -a.get(idx[0]!)!;
    return { rewards, score: a.get(idx[0]!)! };
  }
  for (const i of idx) rewards[i] = -a.get(i)! + idx.filter((j) => j !== i).reduce((s, j) => s + dPair[i]![j]!, 0) / (idx.length - 1);
  const pairs = idx.flatMap((i) => idx.filter((j) => j !== i).map((j) => dPair[i]![j]!));
  return { rewards, score: mean([...a.values()]) - 0.5 * mean(pairs) };
}

// ------------------------------------------------------------------ losses as weights

const sigmoid = (x: number) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

/** DPO as weights (§9): per pair, z, the loss, and the weight of the chosen and rejected replies' trained tokens. */
export function dpo(pairs: readonly { pi_c: number; rho_c: number; pi_r: number; rho_r: number }[], beta = 0.1):
  { z: number; loss: number; chosen_weight: number; rejected_weight: number }[] {
  return pairs.map((p) => {
    const z = beta * ((p.pi_c - p.rho_c) - (p.pi_r - p.rho_r));
    const w = (beta * sigmoid(-z)) / pairs.length;
    return { z, loss: z > -30 ? Math.log1p(Math.exp(-z)) : -z, chosen_weight: w, rejected_weight: -w };
  });
}

/** softmax over a list. */
export function softmax(xs: readonly number[]): number[] {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

/** A decision as weights over its choices' log-probabilities (§9): a known answer, or a target distribution (Brier). */
export function decisionWeights(logprobs: readonly number[], answer: number | null, target: readonly number[] | null = null): { p: number[]; loss: number; weights: number[] } {
  const p = softmax(logprobs);
  if (answer !== null) return { p, loss: -Math.log(p[answer]!), weights: p.map((pk, k) => (k === answer ? 1 : 0) - pk) };
  const t = target!;
  const s = p.reduce((acc, pj, j) => acc + pj * (pj - t[j]!), 0);
  return { p, loss: p.reduce((acc, pk, k) => acc + (pk - t[k]!) ** 2, 0), weights: p.map((pk, k) => -2 * pk * ((pk - t[k]!) - s)) };
}

// ------------------------------------------------------------------ choosing

/** The best (check, temperature): the highest score; the earlier check, then the lower temperature, on a tie. */
export function choose(checks: readonly { step: number; scores: Readonly<Record<string, number | null>> }[]): { step: number; temperature: number; score: number } | null {
  let best: { step: number; temperature: number; score: number } | null = null;
  for (const c of checks) {
    for (const t of Object.keys(c.scores).sort((a, b) => Number(a) - Number(b))) {
      const v = c.scores[t];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      if (best === null || v > best.score) best = { step: c.step, temperature: Number(t), score: v };
    }
  }
  return best;
}
