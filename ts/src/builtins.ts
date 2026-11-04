/**
 * Built-in plugins, made with the public hooks only (contract/plugins.md):
 * anyone can replace them with their own.
 *
 * ```ts
 * const chat = tutor.conversation("alex", { store: "tutoring/", plugins: [compaction({ keep: 20 })] });
 *
 * const researcher = delegate(research, { description: "Look things up in the notes." });
 * const assistant = ai("assistant", { input: { request: t.string() }, tools: [researcher] });
 * ```
 */

import { InferencePricing } from "@lm15/lm15";
import * as calllog from "./calllog.ts";
import type { AnyAIFunction } from "./fn.ts";
import { ai } from "./fn.ts";
import type { AnyModule } from "./module.ts";
import { Plugin, pricing, type BeforeCallHook, type Price, type ShownTurn, type Spent, type StepHook } from "./plugins.ts";
import type { ToolOutput } from "./results.ts";
import { withSettings } from "./settings.ts";
import { t } from "./shapes.ts";
import { effectsOf, type Effects, type Tool } from "./tools.ts";
import { Conversation, type TurnStream } from "./conversations.ts";
import * as lmcc from "lmcc";

type Rec = Record<string, unknown>;

let summarizer: AnyAIFunction | null = null;

function defaultSummarizer(): AnyAIFunction {
  summarizer ??= ai("summarize_conversation", {
    description: "Write the summary of a conversation that whoever continues it needs: what was asked, what was answered and decided, "
      + "names, numbers and facts that came up, and what is still open. Start from the earlier summary (empty when there is none) and "
      + "fold the new turns into it. Be complete about facts and short about everything else.",
    input: { earlier_summary: t.string(), new_turns: t.list(t.record(t.json())) },
    output: t.string(),
  }) as unknown as AnyAIFunction;
  return summarizer;
}

/**
 * Keep a long conversation short: older turns are folded into a summary.
 *
 * When a turn ends and more than `keep + every` turns of its branch are not
 * yet summarized, every turn but the last `keep` is folded into the summary
 * (the earlier summary and those turns, given to `summarize`). The summary is
 * kept in the conversation (an entry at that turn, so each branch has its
 * own), and the next turns are shown it, as a section of the instruction,
 * with only the turns after it. The call that wrote it is in the call log;
 * what a turn was shown is in its record, so a rated turn is asked again
 * with the same summary.
 *
 * `summarize(earlierSummary, newTurns)` gives the summary (default: an AI
 * function of FunctAI's, on `lm` or the configured model). `name`: two
 * compactions with different settings on one conversation need two names.
 */
export function compaction(opts: {
  keep?: number; every?: number; summarize?: (earlierSummary: string, newTurns: Rec[]) => string | Promise<string>; lm?: string; name?: string;
} = {}): Plugin {
  const keep = opts.keep ?? 20;
  const every = opts.every ?? 10;
  if (!Number.isInteger(keep) || keep < 0 || !Number.isInteger(every) || every < 1) {
    throw new RangeError("compaction({ keep, every }): keep a whole number of at least 0, every at least 1");
  }
  const latest = (entries: Rec[]) => (entries.length ? entries[entries.length - 1]!["data"] as Rec : null);
  const plugin = new Plugin(opts.name ?? "compaction", { version: "1.0.0", description: `compaction: summarize all but the last ${keep} turns` });
  plugin.turnEnd(async (event) => {
    if (event.state !== "done") return;
    const turns = event.turns();
    const summary = latest(event.entries("summary"));
    const ids = turns.map((x) => x.id);
    const start = summary && ids.includes(summary["through"] as string) ? ids.indexOf(summary["through"] as string) + 1 : 0;
    const open = turns.slice(start);
    if (open.length < keep + every) return;
    const folded = open.slice(0, open.length - keep);
    const rows = folded.map((x: ShownTurn) => ({ ...x.inputs, ...x.outputs }));
    const earlierSummary = (summary?.["text"] ?? "") as string;
    const text = await withSettings({ caller: { compaction: event.turn } }, async () => {
      if (opts.summarize) return opts.summarize(earlierSummary, rows);
      const fn = opts.lm ? defaultSummarizer().using({ lm: opts.lm }) : defaultSummarizer();
      return fn({ earlier_summary: earlierSummary, new_turns: rows } as never) as Promise<string>;
    });
    if (typeof text !== "string" || !text.trim()) throw new TypeError("a summary is a text");
    await event.remember("summary", { through: folded[folded.length - 1]!.id, text, turns: Number(summary?.["turns"] ?? 0) + folded.length });
  });
  plugin.context((event) => {
    const summary = latest(event.entries("summary"));
    if (!summary) return undefined;
    const branch = event.conversation.branchOf(event.parent);
    const through = summary["through"] as string;
    if (!branch.includes(through)) return undefined;
    const covered = new Set(branch.slice(0, branch.indexOf(through) + 1));
    return {
      keep: event.turns.filter((x) => !covered.has(x.id)).map((x) => x.id),
      sections: [`Earlier in this conversation (${summary["turns"]} turns, summarized):\n${summary["text"]}`],
    };
  });
  return plugin;
}

type Program = (AnyAIFunction | AnyModule) & { conversation?: unknown };

/**
 * Another program as a tool: an assistant hands part of its work to it.
 *
 * Asked inside a conversation's turn, the program answers in a conversation
 * of its own (`<conversation>.<name>`, in the same store), which follows the
 * branch of the turn that asked: asked again later on that branch, it
 * remembers what it was asked before; on another branch, it does not. Its
 * calls are in the asking turn's call tree, under the tool call. Outside a
 * conversation, or with `remember: false`, it is called plainly. `effects`:
 * what it does to the world (default: `"reads"` when every tool it has only
 * reads, else unknown, which counts as `"changes"`).
 */
export function delegate(program: Program, opts: { name?: string; description?: string; remember?: boolean; effects?: Effects | null } = {}): Tool {
  const name = opts.name ?? program.name;
  const iface = program.interface;
  const remember = opts.remember ?? true;
  const run = async (input: Rec): Promise<unknown> => {
    const call = calllog.current.get();
    const turnRun = (call?.turnRun ?? null) as unknown as { conversation: Conversation; turn: string } | null;
    if (!remember || !turnRun) return (program as unknown as (i: unknown) => Promise<unknown>)(input);
    const outer = turnRun.conversation;
    let sub = `${outer.id}.${name}`;
    if (sub.length > 200) sub = `${outer.id.slice(0, 150)}.${lmcc.sha256Hex(sub).slice(0, 16)}`;
    let chat = new Conversation(program as never, sub, { store: outer.store });
    chat._delegated = true;
    await outer.readLog();
    const last = outer.entries("delegate", name, { branch: turnRun.turn });
    if (last.length) {
      chat = chat.continueFrom((last[last.length - 1]!["data"] as Rec)["turn"] as string);
      chat._delegated = true;
    } else {
      chat._head = null;                              // the first delegation on this branch starts afresh
      chat._exact = true;
    }
    const s: TurnStream = chat.stream(input);
    const value = await s.result;
    const turn = await s.turn;
    await outer.remember("delegate", name, { turn: turn.id }, { turn: turnRun.turn });
    return value;
  };
  return {
    name, description: opts.description ?? (iface.description || `Ask ${program.name}.`),
    // the tool's inputs are the program's, as its interface states them
    parameters: {
      type: "object", properties: Object.fromEntries(iface.inputs.map((f) => [f.name, f.shape])),
      required: iface.inputs.filter((f) => !f.optional).map((f) => f.name),
    },
    run, effects: opts.effects !== undefined ? opts.effects : effectsOf(program),
    _program: program,                               // a setup counts the program it stands for (calls.md, *Setups*)
  } as Tool;
}

// ------------------------------------------------------------------ long tool loops (the step hook)

export const REACHED = (what: string) =>
  `This task has reached its limit (${what}). Do not ask for any more tools: give your answer now, from what you have found so far.`;
export const LAST_STEP = "This is the last step this task may take. Do not ask for any more tools: give your answer now, from what you have found so far.";
export const PICTURE_GONE = "(A picture was here. It is no longer shown, to save space.)";

const numberText = (x: number) => Number.isInteger(x) ? String(x) : String(x);

/** Options of `limits`: at least one of `tokens`, `cost`, `seconds`; `cost` needs `prices`. */
export interface LimitsOptions {
  /** Every token read and written (`total_tokens`), summed over the tree. */
  readonly tokens?: number;
  /** The list price of the tree's requests, at `prices`. Exact or nothing: an unknown cost counts as reached. */
  readonly cost?: number;
  /** Since the tree's outermost call started. */
  readonly seconds?: number;
  /** A model's name (as you ask for it) → lm15's `InferencePricing` (or its JSON form). FunctAI ships no prices. */
  readonly prices?: Readonly<Record<string, Price>>;
  /** `"answer"` (the default): a tool loop at a limit, or at its last step, is told to answer with what it has; `"stop"`: the call stops. */
  readonly then?: "answer" | "stop";
  readonly name?: string;
}

/** Which limits a tree's spending reached: each `[limit, why]` (why: when its count is not known); `[]` when none. */
export function reached(spent: Spent, o: LimitsOptions): [string, string | null][] {
  const out: [string, string | null][] = [];
  if (o.tokens !== undefined) {
    const got = spent.tokens;
    if (got === null) out.push([`${numberText(o.tokens)} tokens`, "a request did not say how many it used"]);
    else if (got >= o.tokens) out.push([`${numberText(o.tokens)} tokens`, null]);
  }
  if (o.cost !== undefined) {
    const got = spent.cost(o.prices ?? {});
    if (got === null) {
      const unknown = [...new Set(spent.exchanges.map(([m]) => m).filter((m) => pricing(o.prices ?? {}, m) === null))].sort();
      out.push([`cost ${numberText(o.cost)}`, unknown.length ? `no price is known for ${unknown.join(", ")}`
        : "a request's cost is not known from its token counts (a price without cache_read_per_million and cache_write_per_million is exact only for requests that report their cache counts; give both, equal to input_per_million when the provider does not charge them apart)"]);
    } else if (got >= o.cost) out.push([`cost ${numberText(o.cost)}`, null]);
  }
  if (o.seconds !== undefined && spent.seconds >= o.seconds) out.push([`${numberText(o.seconds)} seconds`, null]);
  return out;
}

/** lm15's `InferencePricing` as JSON (plugins.md, *limits*): its fields by lm15's names, without those at lm15's defaults (an unset rate, currency USD, no tiers). */
function priceJson(p: InferencePricing): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  // lm15's JSON form, as plain numbers (its RawNumber keeps "3.0"; canonical JSON writes 3 either way)
  const plain = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(plain);
    if (typeof v !== "object" || v === null) return v;
    if (typeof (v as { raw?: unknown }).raw === "string" && Object.keys(v).length === 1) return Number((v as { raw: string }).raw);
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, plain(x)]));
  };
  for (const [k, v] of Object.entries(plain(InferencePricing.toJSON(p)) as Record<string, unknown>)) {
    if (v === undefined || v === null || (k === "currency" && v === "USD") || (k === "tiers" && Array.isArray(v) && !v.length)) continue;
    out[k] = v;
  }
  return out;
}

const stoppedBy = (hit: readonly [string, string | null][]) =>
  "limit reached: " + hit.map(([l, why]) => why ? `${l} (${why})` : l).join("; ");

/**
 * Stop a call tree (an agent, a pipeline) when it has used enough tokens,
 * money or time (contract/plugins.md, "limits"). Nothing is limited by
 * default; this plugin is opt-in. Counted over the whole tree a call starts,
 * from the requests a provider answered in this process; checked before each
 * AI function's call and before each step of a tool loop, so one request may
 * go past a limit; the next does not start.
 *
 * ```ts
 * const prices = { "claude-opus-5-5": { inputPerMillion: 5, outputPerMillion: 25, cacheReadPerMillion: 0.5, cacheWritePerMillion: 6.25 } };
 * await withSettings({ plugins: [limits({ cost: 3, seconds: 600, prices })] }, () => conceptTest(url));
 * ```
 */
export function limits(o: LimitsOptions): Plugin {
  const then = o.then ?? "answer";
  if (then !== "answer" && then !== "stop") throw new TypeError(`then is "answer" or "stop", not ${JSON.stringify(then)}`);
  if (o.cost !== undefined && !o.prices) throw new TypeError("a cost limit needs prices: { model: InferencePricing }; FunctAI ships none");
  for (const [label, v] of [["tokens", o.tokens], ["cost", o.cost], ["seconds", o.seconds]] as const) {
    if (v !== undefined && (typeof v !== "number" || !(v > 0))) throw new TypeError(`${label} is a number above 0`);
  }
  if (o.tokens === undefined && o.cost === undefined && o.seconds === undefined) throw new TypeError("limits needs at least one of tokens, cost, seconds");
  for (const model of Object.keys(o.prices ?? {})) pricing(o.prices!, model);    // a malformed price is refused now, not mid-run
  // its settings as JSON (plugins.md, *limits*): a program's setup counts them
  const config: Record<string, unknown> = {};
  for (const k of ["tokens", "cost", "seconds"] as const) if (o[k] !== undefined) config[k] = o[k];
  if (o.prices && Object.keys(o.prices).length) {
    config["prices"] = Object.fromEntries(Object.keys(o.prices).map((m) => [m, priceJson(pricing(o.prices!, m)!)]));
  }
  config["then"] = then;
  return new Plugin(o.name ?? "limits", { version: "1.0.0", description: "stop a call tree when it used enough tokens, money or time", config })
    .beforeCall((call: BeforeCallHook) => {
      let hit = reached(call.spent, o);
      if (!hit.length && o.cost !== undefined && typeof call.lm === "string" && pricing(o.prices ?? {}, call.lm) === null) {
        hit = [[`cost ${numberText(o.cost)}`, `no price is known for ${call.lm}`]];
      }
      return hit.length ? { stop: stoppedBy(hit) } : undefined;
    })
    .step((s: StepHook) => {
      const hit = reached(s.spent, o);
      if (hit.length && then === "stop") return { stop: stoppedBy(hit) };
      if (then === "answer" && s.tools.length && (hit.length || s.lastStep)) {
        return { tools: [], reminder: hit.length ? REACHED(hit.map(([l]) => l).join(", ")) : LAST_STEP };
      }
      return undefined;
    });
}

/**
 * Which earlier tool results `keepPictures` shows otherwise, and as what: the
 * pictures before the newest `last` are replaced by a short text, once at
 * least `atOnce` of them are due.
 */
export function picturesToDrop(results: readonly { invocation: number; output: ToolOutput }[], last: number, atOnce = 1): Record<string, Rec[]> {
  let seen = 0;
  const drop = new Map<number, Set<number>>();
  for (let r = results.length - 1; r >= 0; r--) {
    const out = results[r]!.output;
    if (typeof out === "string") continue;
    for (let i = out.length - 1; i >= 0; i--) {
      if (out[i]!["type"] !== "image") continue;
      seen++;
      if (seen > last) {
        const inv = results[r]!.invocation;
        if (!drop.has(inv)) drop.set(inv, new Set());
        drop.get(inv)!.add(i);
      }
    }
  }
  if ([...drop.values()].reduce((n, s) => n + s.size, 0) < atOnce) return {};
  const replace: Record<string, Rec[]> = {};
  for (const r of results) {
    const gone = drop.get(r.invocation);
    if (!gone || typeof r.output === "string") continue;
    replace[String(r.invocation)] = r.output.map((p, i) => gone.has(i) ? { type: "text", text: PICTURE_GONE } : { ...p });
  }
  return replace;
}

/**
 * Show a long tool loop only its newest pictures: before each step, the
 * pictures tool results gave earlier than the newest `last` are replaced by
 * a short text (their own text stays). `atOnce`: drop old pictures only once
 * this many are due, together, so a provider's prompt cache keeps the start
 * of the request longer. The turn and the call log keep every picture.
 *
 * ```ts
 * const agent = browse.using({ plugins: [keepPictures(3)], maxSteps: 200 });
 * ```
 */
export function keepPictures(last = 3, opts: { atOnce?: number; name?: string } = {}): Plugin {
  const atOnce = opts.atOnce ?? 1;
  if (!Number.isInteger(last) || last < 0) throw new TypeError("last is how many of the newest pictures stay shown (0 or more)");
  if (!Number.isInteger(atOnce) || atOnce < 1) throw new TypeError("atOnce is 1 or more");
  return new Plugin(opts.name ?? "keep-pictures", { version: "1.0.0", description: "show a tool loop only its newest pictures",
    config: { last, at_once: atOnce } })
    .step((s: StepHook) => {
      const replace = picturesToDrop(s.results, last, atOnce);
      return Object.keys(replace).length ? { replace } : undefined;
    });
}
