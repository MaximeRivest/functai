/**
 * Tools that say what they do, and asking a person before they run
 * (contract/tools.md).
 *
 * ```ts
 * const readNote = tool("read_note", { description: "The text of one note.", input: { name: t.string() }, effects: "reads" },
 *   ({ name }) => notes[name]);
 * const writeNote = tool("write_note", { description: "Replace a note's text.", input: { name: t.string(), text: t.string() },
 *   effects: "changes" }, ({ name, text }) => { notes[name] = text; return "saved"; });
 *
 * const gardener = ai("gardener", { input: { request: t.string() }, tools: [readNote, writeNote] });
 * await gardener("Merge groceries into todo", { approve: async (a) => confirm(`${a.path}(${JSON.stringify(a.input)})?`) });
 * const chat = gardener.conversation("notes", { store: "notes-chats/", approve: "changes" });
 * await chat("Tidy todo");             // rejects with Waiting; the turn is saved as waiting
 * await chat.head!.approve();          // from any process: the turn goes on
 * ```
 *
 * A tool that says nothing has unknown effects, which every rule treats as
 * `"changes"`: forgetting to declare is safe. `approve` is a setting (in
 * `configure`, `using`, `ai()`, a conversation, a turn): a function asked at
 * once, or a rule a person answers later. No approval by default.
 */

import { Context } from "./host.ts";
import { Cancelled, ApprovalError, type Approval } from "./errors.ts";
import type { Call } from "./calllog.ts";

type Rec = Record<string, unknown>;

/** What a tool does to the world: it only looks, or it changes something (writes, sends, pays). */
export type Effects = "reads" | "changes";

/** A tool the model may call: its name, what it does, the JSON Schema of its input, its code, and its effects. */
export interface Tool {
  readonly name: string;
  readonly description?: string;
  readonly parameters: Rec;
  readonly run: (input: any) => unknown;
  /** `"reads"`, `"changes"`, or unset: unknown, which every approval rule treats as `"changes"`. */
  readonly effects?: Effects | null;
}

/**
 * Who answers a question about a tool call: `true` (yes), `false` (no), or a
 * reason to refuse (a text), sync or async. Asked about each tool call the
 * `"changes"` rule selects.
 */
export type ApproveFunction = (approval: Approval) => boolean | string | null | undefined | PromiseLike<boolean | string | null | undefined>;

/**
 * The `approve` setting: a function asked at once, or a rule a person answers
 * later: `"changes"` (tools that change things or say nothing), `"all"`, or a
 * list of tool names, approval paths (`"support/answer/refund"`) or their
 * endings (`"answer/refund"`).
 */
export type ApproveSetting = ApproveFunction | "changes" | "all" | readonly string[];

/** What the model is shown for a refused tool call. */
export const DENIED = "The person did not allow this call.";

/** What a tool says it does: its `effects`; an AI function used as a tool reads, unless one of its own tools changes things (or says nothing). */
export function effectsOf(tool: unknown): Effects | null {
  const e = (tool as { effects?: unknown } | null)?.effects;
  if (e === "reads" || e === "changes") return e;
  const inner = (tool as { _aiTools?: readonly unknown[] } | null)?._aiTools;
  if (Array.isArray(inner)) return inner.every((t) => effectsOf(t) === "reads") ? "reads" : null;
  return null;
}

/** Refuse, where it is set, an `approve` value that is neither a function nor a rule. */
export function checkApprove(value: unknown, where: string): void {
  if (value === undefined || value === null || typeof value === "function") return;
  if (value === "changes" || value === "all") return;
  if (Array.isArray(value) && value.every((v) => typeof v === "string" && v)) return;
  throw new TypeError(`${where}: approve is a function, "changes", "all", or a list of tool names and approval paths`);
}

/** A tool call's approval path: the names of the calls from the outermost to the one that asked, then the tool's (`support/answer/refund`). */
export function pathOf(call: Call, name: string): string {
  return [...call.path.split("/").filter(Boolean).map((p) => p.split("#")[0]!), name].join("/");
}

/**
 * Whether a rule asks a person about this tool call (tools.md): `"changes"`
 * (and a function, which is asked what `"changes"` asks) for a tool that
 * changes things or says nothing; `"all"` for every one; a list for a tool
 * named in it, or whose path is in it or ends with `/<entry>`.
 */
export function asks(rule: unknown, approval: Pick<Approval, "name" | "path" | "effects">): boolean {
  if (rule === undefined || rule === null) return false;
  if (typeof rule === "function" || rule === "changes") return approval.effects !== "reads";
  if (rule === "all") return true;
  for (const entry of rule as readonly string[]) {
    if (entry === approval.name || entry === approval.path || approval.path.endsWith("/" + entry)) return true;
  }
  return false;
}

/** (allowed, reason) from what a function answered. */
export function verdictOf(answer: unknown): [boolean, string | null] {
  if (answer === true) return [true, null];
  if (answer === false || answer === null || answer === undefined) return [false, null];
  if (typeof answer === "string") return [false, answer];
  throw new TypeError("an approve function answers true, false, or a reason to refuse (a text)");
}

/** What the model is shown for a refused tool call (tools.md, "A refusal is an answer the model sees"). */
export function denial(reason: string | null | undefined): string {
  return DENIED + (reason ? ` Reason: ${reason}` : "");
}

/** Whom an approval is addressed to: the owner, or the caller when a served program lets its caller answer (serving.md). */
export const APPROVALS_TO = new Context<"owner" | "caller">();

// ------------------------------------------------------------------ waiting in this process (a stream)

interface Pending {
  readonly approval: Approval;
  readonly answer: (allowed: boolean, reason: string | null, by: string | null) => void;
}

/** Tool calls of this process waiting for an answer, by `call|invocation|plugin`. */
const pending = new Map<string, Pending>();
const keyOf = (call: string, invocation: number, plugin: string) => `${call}|${invocation}|${plugin}`;

/** The approvals calls of this process wait for (on streams). */
export function waitingHere(): Approval[] {
  return [...pending.values()].map((p) => p.approval);
}

/** Answer an approval a call of this process waits for. */
export function answerHere(call: string, invocation: number, allowed: boolean, opts: { reason?: string | null; by?: string | null; plugin?: string } = {}): void {
  const k = keyOf(call, invocation, opts.plugin ?? "approval");
  const p = pending.get(k);
  if (!p) throw new Error(`no tool call waits here for invocation ${invocation} of call ${call}`);
  pending.delete(k);
  p.answer(allowed, opts.reason ?? null, opts.by ?? null);
}

function waitHere(call: Call, approval: Approval): Promise<[boolean, string | null, string | null]> {
  return new Promise((resolve, reject) => {
    const k = keyOf(approval.call, approval.invocation, approval.plugin);
    const signal = call.signal;
    const stop = () => {
      pending.delete(k);
      reject(new Cancelled());
    };
    if (signal?.aborted) return stop();
    signal?.addEventListener("abort", stop, { once: true });
    pending.set(k, {
      approval,
      answer: (allowed, reason, by) => {
        signal?.removeEventListener("abort", stop);
        resolve([allowed, reason, by]);
      },
    });
  });
}

/** Whether a stream reads this call (a person can be asked through it). */
function hasStream(call: Call): boolean {
  return (call.node?.streams.length ?? 0) > 0;
}

/**
 * Ask whether a tool call may run (a plugin's `ask`): `[allowed, reason
 * refused, by whom]`. `decide` answers in place of a person. A turn being
 * resumed takes the answer recorded for it. Shows `approval`, then
 * `approved`. Otherwise: in a conversation the turn waits (it throws: the
 * turn is saved as waiting); on a stream the call waits for `s.approve()`; a
 * plain call refuses `approval-required`.
 */
export async function askPerson(call: Call, approval: Approval, decide?: ApproveFunction | null): Promise<[boolean, string | null, string | null]> {
  const run = call.turnRun;
  if (run) {
    const known = run.recordedApproval(call, approval);
    if (known) {
      const [allowed, reason, by, fresh] = known;
      if (fresh) {                                    // answered while the turn waited: the log says so now
        call.node?.log.frontier();
        call.event("approved", { id: approval.id, invocation: approval.invocation, plugin: approval.plugin, verdict: allowed ? "yes" : "no", by, reason });
      }
      return [allowed, reason, by];
    }
  }
  call.node?.log.frontier();
  call.event("approval", {
    id: approval.id, invocation: approval.invocation, name: approval.name, input: approval.input, effects: approval.effects,
    path: approval.path, to: APPROVALS_TO.get() ?? "owner", plugin: approval.plugin, ...(approval.question ? { question: approval.question } : {}),
  });
  let allowed: boolean, reason: string | null, by: string | null;
  if (decide) {
    [allowed, reason] = verdictOf(await decide(approval));
    by = null;
  } else if (run) {
    run.pause(call, approval);                          // throws: the turn waits, saved
  } else if (hasStream(call)) {
    [allowed, reason, by] = await waitHere(call, approval);
  } else {
    throw new ApprovalError(`${approval.path}: this tool call needs a person's answer (${approval.plugin}), and a plain call has nobody to ask. `
      + "Give approve a function, stream the call and answer with s.approve(), or use a conversation, where the turn waits.", approval);
  }
  call.event("approved", { id: approval.id, invocation: approval.invocation, plugin: approval.plugin, verdict: allowed! ? "yes" : "no", by: by!, reason: reason! });
  run?.noteApproval(call, approval, allowed!, reason!, by!);
  return [allowed!, reason!, by!];
}
