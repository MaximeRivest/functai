/**
 * The errors FunctAI raises beyond lmcc's refusals, each with the contract's
 * `code` when it has one (contract/README.md, "Refusal codes FunctAI
 * defines").
 */

type Rec = Record<string, unknown>;

/** A stream was closed, a signal aborted, or a turn was stopped: the call ended without an answer. */
export class Cancelled extends Error {
  constructor(message = "the stream was closed") {
    super(message);
    this.name = "Cancelled";
  }
}

/** An error with one of the contract's codes. */
export class FunctAIError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = new.target.name;
  }
}

/**
 * A conversation refused (contract/conversations.md): `conversation-id`,
 * `conversation-content`, `conversation-opaque`, `conversation-signature`,
 * `conversation-busy`, `conversation-nested`, `turn-unknown`, `turn-state`,
 * `turn-unfinished`, or a store's `store-conflict`. `turn` names the turn
 * when there is one.
 */
export class ConversationError extends FunctAIError {
  readonly turn: string | null;
  constructor(code: string, message: string, opts: { turn?: string | null } = {}) {
    super(code, message);
    this.turn = opts.turn ?? null;
  }
}

/** One tool call waiting for a person's answer (contract/tools.md). */
export interface Approval {
  /** The id of the call that asked for the tool (an AI function's call). */
  readonly call: string;
  /** The tool call's number among the tool calls of that call (1, 2, …). */
  readonly invocation: number;
  /** The id the model gave the tool call. */
  readonly id: string;
  readonly name: string;
  readonly input: unknown;
  /** What the tool says it does: `"reads"`, `"changes"`, or null (unknown: it counts as `"changes"`). */
  readonly effects: "reads" | "changes" | null;
  /** Where it is, by names: `support/answer/refund`. Rules are written with it. */
  readonly path: string;
  /** The asking call's place in its tree (`support#1/answer#1`): what resuming finds it by. */
  readonly site: string;
  /** The plugin that asks (`approval`: the `approve` setting). */
  readonly plugin: string;
  /** Why it asks, when it says. */
  readonly question?: string | null;
}

/**
 * A conversation's turn stopped to wait for a person's answer (code
 * `turn-waiting`): its turn is saved as `waiting`; answer with
 * `err.turn.approve()` or `err.turn.deny(reason)`, from this process or any
 * other that opens the conversation.
 */
export class Waiting extends FunctAIError {
  readonly turn: unknown;
  readonly approvals: readonly Approval[];
  constructor(message: string, turn: unknown, approvals: readonly Approval[]) {
    super("turn-waiting", message);
    this.turn = turn;
    this.approvals = approvals;
  }
}

/**
 * A tool call needs a person's answer and nobody can be asked
 * (`approval-required`): a plain call. Give `approve` a function, stream the
 * call (and answer with `s.approve()`), or use a conversation.
 */
export class ApprovalError extends FunctAIError {
  readonly approval: Approval;
  constructor(message: string, approval: Approval) {
    super("approval-required", message);
    this.approval = approval;
  }
}

/**
 * A plugin refused or failed (contract/plugins.md): `plugin-api`,
 * `plugin-hook`, `plugin-name`, `plugin-change`, `plugin-failed`. `plugin`
 * and `hook` name where.
 */
export class PluginError extends FunctAIError {
  readonly plugin: string | null;
  readonly hook: string | null;
  constructor(code: string, message: string, opts: { plugin?: string | null; hook?: string | null; cause?: unknown } = {}) {
    super(code, message);
    this.plugin = opts.plugin ?? null;
    this.hook = opts.hook ?? null;
    if (opts.cause !== undefined) (this as { cause?: unknown }).cause = opts.cause;
  }
}

/** A program cannot be served (contract/serving.md): `serve-opaque`, `serve-keys`. */
export class ServeError extends FunctAIError {}

/** A served program's server answered with an error: `code` is its code (or `remote-<status>`), `status` the HTTP status. */
export class RemoteError extends FunctAIError {
  readonly status: number;
  readonly type: string;
  constructor(code: string, message: string, opts: { status?: number; type?: string } = {}) {
    super(code, message);
    this.status = opts.status ?? 0;
    this.type = opts.type ?? "";
  }
}

/**
 * A bake, or a call of a baked model, refused (contract/baked.md):
 * `baked-fixed`, `baked-derived`, `baked-changed`, `baked-format`, `bake-rows`.
 */
export class BakeError extends FunctAIError {}

/** An error as plain data, for records: its type, its code, its message. */
export function errorInfo(err: unknown): Rec {
  const e = err as { name?: string; code?: unknown; message?: string } | null | undefined;
  const out: Rec = { type: typeof e?.name === "string" ? e.name : "Error" };
  if (typeof e?.code === "string") out["code"] = e.code;
  out["message"] = typeof e?.message === "string" ? e.message : String(err);
  return out;
}
