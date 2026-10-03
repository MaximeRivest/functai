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

import * as calllog from "./calllog.ts";
import type { AnyAIFunction } from "./fn.ts";
import { ai } from "./fn.ts";
import type { AnyModule } from "./module.ts";
import { Plugin, type ShownTurn } from "./plugins.ts";
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
  };
}
