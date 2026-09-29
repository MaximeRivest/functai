/**
 * A call tree's events as data (contract/streaming.md, format 2): their
 * form, replaying and resuming them, following a log live, the kept form,
 * the rules a store keeps, and which observers and journal a tree gets.
 * Nothing here runs a call; `log.ts` makes the events.
 *
 * ```ts
 * const reader = new Follower({ form: "kept" });
 * for await (const e of source) reader.receive(e);   // "kept", "duplicate", "stale", "rewind", "loss", …
 * reader.state(tree);                                // { calls: { [id]: { ended, fields } }, finished }
 * ```
 */

import * as lmcc from "lmcc";
import { passes } from "./schema.ts";
import { getOwn, setOwn } from "./values.ts";

type Rec = Record<string, unknown>;

// ------------------------------------------------------------------ the events

/** An event of a log, named by the writer that numbered it and its seq (with the log's tree, one event in every form). */
export interface Position {
  readonly writer: number;
  readonly seq: number;
}

/** What every event says: which log it is in and where, when, and which call it is about. */
type Envelope<K extends string> = {
  readonly functai_event: 2;
  readonly kind: K;
  /** The log's id: the id of the tree's outermost call. */
  readonly tree: string;
  /** The writer that numbered it: 1, or the number a later writer's claim gave. */
  readonly writer: number;
  /** Its place in the log (a later writer may use again the seq of an event never kept: `writer` tells them apart). */
  readonly seq: number;
  /** The position of the event before it in the form being read (null for the form's first). */
  readonly after: Position | null;
  /** When the writer numbered it (RFC 3339 UTC, six fraction digits; never less than the one before). */
  readonly at: string;
  /** The call it is about. */
  readonly call: string;
  /** That call's program's name. */
  readonly function: string;
};

/** The call log's `program` object (calls.md). */
export type ProgramInfo = {
  readonly name: string;
  readonly kind: "ai" | "module";
  readonly module: string;
  readonly version: string;
  readonly signature?: string;
  readonly interface: string;
  readonly answer: string;
  readonly saved?: string;
  readonly file?: string;
  readonly line?: number;
};

/** A call a call was shown as context (calls.md, "Saw"). */
export type SawEntry = { readonly call: string; readonly steps?: true; readonly without?: readonly string[]; readonly slot?: string }
  | { readonly saw_of: string } | Readonly<Rec>;

/** An error as events and records hold it. */
export type ErrorInfo = {
  readonly type: string;
  readonly message?: string;
  readonly code?: string;
};

export type StartedEvent = Envelope<"started"> & {
  readonly parent: string | null;
  readonly root: string;
  readonly program: ProgramInfo;
  /** Absent in the kept form when no input is kept. */
  readonly inputs?: Rec;
  /** False in a form that left values out; `omitted` then names them. */
  readonly content: boolean;
  readonly omitted?: { readonly inputs: readonly string[]; readonly outputs: readonly string[] };
  readonly saw: readonly SawEntry[];
};
/** A request to a model begins: it empties the call's fields (law 3). Its `request` events are the call record's exchanges (law 8). */
export type RequestEvent = Envelope<"request"> & { readonly request: number; readonly model: string | null };
/** A piece of an output's text, exact (law 2). */
export type TextEvent = Envelope<"text"> & { readonly field: string; readonly answer: boolean; readonly text: string };
export type ThinkingEvent = Envelope<"thinking"> & { readonly text: string };
export type ToolCallEvent = Envelope<"tool_call"> & { readonly id: string; readonly name: string; readonly input?: unknown; readonly content?: false };
export type ToolResultEvent = Envelope<"tool_result"> & { readonly id: string; readonly name: string; readonly output?: string; readonly content?: false };
/** The model is asked again: empties the call's fields at once. */
export type RetryEvent = Envelope<"retry"> & { readonly reason?: string; readonly wait: number | null; readonly content?: false };
export type DoneEvent = Envelope<"done"> & { readonly value?: unknown; readonly content?: false };
export type FailedEvent = Envelope<"failed"> & { readonly error: ErrorInfo; readonly content?: false };

/** An event of a call tree's log (streaming.md, format 2). */
export type StreamEvent = StartedEvent | RequestEvent | TextEvent | ThinkingEvent | ToolCallEvent | ToolResultEvent
  | RetryEvent | DoneEvent | FailedEvent;

/** Any object read as an event: a known kind, or one a later writer added (readers skip what they do not know). */
export type EventLike = Readonly<Rec> & { readonly kind?: unknown };

const KINDS = new Set(["started", "request", "text", "thinking", "tool_call", "tool_result", "retry", "done", "failed"]);
const ENVELOPE = ["functai_event", "kind", "tree", "writer", "seq", "after", "at", "call", "function"];
const KEYS: Record<string, readonly string[]> = {
  started: ["parent", "root", "program", "inputs", "content", "omitted", "saw"], request: ["request", "model"],
  text: ["field", "answer", "text"], thinking: ["text"], tool_call: ["id", "name", "input", "content"],
  tool_result: ["id", "name", "output", "content"], retry: ["reason", "wait", "content"], done: ["value", "content"],
  failed: ["error", "content"],
};

/** An event's own position. */
export const positionOf = (e: EventLike): Position => ({ writer: e["writer"] as number, seq: e["seq"] as number });
/** Whether two positions (or nulls) are the same: both numbers. */
export const samePosition = (a: Position | null | undefined, b: Position | null | undefined): boolean =>
  (a ?? null) === null ? (b ?? null) === null : (b ?? null) !== null && a!.writer === b!.writer && a!.seq === b!.seq;

// ------------------------------------------------------------------ replaying

/** What a watcher of a form saw of one call: whether it ended, and each field's text so far. */
export interface CallState {
  ended: null | "done" | "failed";
  fields: Record<string, string>;
}

/** What a watcher of a form saw after an event: the calls started so far, and whether the log is finished. */
export interface LogState {
  calls: Record<string, CallState>;
  finished: boolean;
}

/** Whether an object is an event of the format this reader knows (2): what its numbers mean is known. */
export const knownFormat = (e: unknown): boolean =>
  typeof e === "object" && e !== null && (e as Rec)["functai_event"] === 2;

/**
 * Replaying a form of a log (streaming.md, "Replaying"): `started` adds a
 * call; `request` and `retry` empty its fields; `text` appends; `done` and
 * `failed` end it. A kind it does not know changes nothing; an event of a
 * format it does not know stops it (`stopped`): it applies nothing more.
 */
export class Replay {
  private calls: Record<string, CallState> = {};
  private tree: string | null = null;
  finished = false;
  /** Set once an event of a format it does not know came: nothing after it was applied. */
  stopped = false;

  apply(e: EventLike): this {
    if (this.stopped) return this;
    if (!knownFormat(e)) {
      this.stopped = true;
      return this;
    }
    const kind = e.kind as string;
    const c = e["call"] as string;
    this.tree ??= (e["tree"] as string) ?? null;
    if (!KINDS.has(kind)) return this;
    if (kind === "started") {
      setOwn(this.calls, c, { ended: null, fields: {} });
      return this;
    }
    const call = getOwn(this.calls, c);
    if (!call) return this;                         // its started is not in this form: nothing to apply it to
    if (kind === "request" || kind === "retry") call.fields = {};
    else if (kind === "text") {
      const field = e["field"] as string;
      setOwn(call.fields, field, (getOwn(call.fields, field) ?? "") + (e["text"] as string));
    } else if (kind === "done" || kind === "failed") {
      call.ended = kind;
      if (c === this.tree) this.finished = true;
    }
    return this;
  }

  get state(): LogState {
    return { calls: structuredClone(this.calls), finished: this.finished };
  }
}

/**
 * The state after each event of a form, and whether the log is finished.
 * `stopped` is there (true) when an event of a format this reader does not
 * know came: nothing from it on was applied.
 */
export function replay(events: readonly EventLike[]): { views: { calls: Record<string, CallState> }[]; finished: boolean; stopped?: true } {
  const r = new Replay();
  const views = events.map((e) => ({ calls: r.apply(e).state.calls }));
  return { views, finished: r.finished, ...(r.stopped ? { stopped: true as const } : {}) };
}

/** The state after every event of a form. */
export function stateOf(events: readonly EventLike[]): LogState {
  const r = new Replay();
  for (const e of events) r.apply(e);
  return r.state;
}

/** What a source can answer a read. */
export type ReadAnswer<E = StreamEvent> = { readonly events: readonly E[] } | { readonly refuses: "event-unknown" };

/** What a source holding these events (one form of one log) gives a reader that has events up to `after` (null: all). */
export function resume<E extends EventLike>(events: readonly E[], after: Position | null): ReadAnswer<E> {
  if (after === null) return { events: [...events] };
  const i = events.findIndex((e) => samePosition(positionOf(e), after));
  return i < 0 ? { refuses: "event-unknown" } : { events: events.slice(i + 1) };
}

// ------------------------------------------------------------------ following

/** What a follower did with an event it received. */
export type FollowResult = "kept" | "duplicate" | "stale" | "rewind" | "loss" | "unknown-format";

/** Where a reader reads events again: a store, or the process of a writer (`writer`). */
export interface EventSource {
  /** The writer whose process this is; absent for a store. */
  readonly writer?: number;
  read(tree: string, after: Position | null): ReadAnswer | Promise<ReadAnswer>;
}

interface Followed {
  held: { event: EventLike; from: number | "store" }[];
  last: Position | null;
}

/**
 * A reader following one form of logs live (streaming.md, "Following a
 * log"), keeping for each tree the events it holds. `form`: `"kept"` (the
 * kept form, or a view made from it) or `"live"` (the whole log, or a view
 * of it that may show values the kept form lacks); it decides where the
 * reader may resume in place.
 */
export class Follower {
  readonly form: "kept" | "live";
  private readonly trees = new Map<string, Followed>();
  /** Set once an event of a format it does not know arrived: it follows that source no more. */
  stopped = false;

  constructor(opts: { form?: "kept" | "live" } = {}) {
    this.form = opts.form ?? "live";
  }

  /**
   * Take one event, as received. `from`: which process gave it (a writer's
   * number), or `"store"`; a live reader's default is the event's writer.
   */
  receive(e: EventLike, opts: { from?: number | "store" } = {}): FollowResult {
    if (this.stopped || !knownFormat(e)) {
      this.stopped = true;
      return "unknown-format";
    }
    const tree = e["tree"] as string;
    let t = this.trees.get(tree);
    if (!t) this.trees.set(tree, t = { held: [], last: null });
    const writer = t.last?.writer ?? 0;
    const w = e["writer"] as number, seq = e["seq"] as number, after = (e["after"] ?? null) as Position | null;
    if (w < writer) return "stale";
    if (w === writer && seq <= t.last!.seq) return "duplicate";
    let result: FollowResult;
    if (samePosition(after, t.last)) result = "kept";
    else {
      const at = after === null ? -1 : t.held.findIndex((h) => samePosition(positionOf(h.event), after));
      if (w > writer && (after === null || at >= 0)) {
        t.held = t.held.slice(0, at + 1);
        result = "rewind";
      } else return "loss";
    }
    t.held.push({ event: e, from: opts.from ?? (this.form === "live" ? w : "store") });
    t.last = positionOf(e);
    return result;
  }

  /** Let a tree go (a finished log a page no longer shows): it keeps nothing of it. */
  forget(tree: string): void {
    this.trees.delete(tree);
  }

  /** The last event it holds of a tree (null: none). */
  last(tree: string): Position | null {
    return this.trees.get(tree)?.last ?? null;
  }

  /** The events it holds of a tree, in order. */
  held(tree: string): EventLike[] {
    return (this.trees.get(tree)?.held ?? []).map((h) => h.event);
  }

  /** The replay of what it holds: of one tree, or of every tree it follows. */
  state(tree: string): LogState;
  state(): Record<string, LogState>;
  state(tree?: string): LogState | Record<string, LogState> {
    if (tree !== undefined) return stateOf(this.held(tree));
    return Object.fromEntries([...this.trees.keys()].map((k) => [k, stateOf(this.held(k))]));
  }

  /**
   * Whether it may resume in place from `source` (streaming.md, "Resuming"):
   * when the source can give every event it holds as it holds it. Always
   * for the kept form; for a live form, only from the process that gave it
   * every event it holds.
   */
  inPlace(tree: string, source: EventSource): boolean {
    if (this.form === "kept") return true;
    const from = source.writer;
    return from !== undefined && (this.trees.get(tree)?.held ?? []).every((h) => h.from === from);
  }

  /**
   * Read a tree again from a source (after a loss, a reconnection): after its
   * last event when it may resume in place, and from the beginning when it
   * may not or the source answers `event-unknown` (dropping what it held).
   * Returns the reads it made, in order. A source it can read lets a stopped
   * reader follow again; an event of a format it does not know stops it
   * there (it takes nothing from that event on).
   */
  async recover(tree: string, source: EventSource): Promise<{ after: Position | null; answer: ReadAnswer }[]> {
    this.stopped = false;
    let t = this.trees.get(tree);
    if (!t) this.trees.set(tree, t = { held: [], last: null });
    const reads: { after: Position | null; answer: ReadAnswer }[] = [];
    let answer: ReadAnswer | null = null;
    if (this.inPlace(tree, source)) {
      answer = await source.read(tree, t.last);
      reads.push({ after: t.last, answer });
    }
    if (answer === null || "refuses" in answer) {
      answer = await source.read(tree, null);
      reads.push({ after: null, answer });
      t.held = [];
      t.last = null;
    }
    if ("events" in answer) {
      const from = source.writer ?? "store";
      for (const e of answer.events) {
        if (!knownFormat(e)) {
          this.stopped = true;
          break;
        }
        t.held.push({ event: e, from });
        t.last = positionOf(e);
      }
    }
    return reads;
  }
}

// ------------------------------------------------------------------ the kept form

/** Which fields of a call its `logContent` keeps. */
export interface KeptFields {
  readonly inputs: Readonly<Record<string, boolean>>;
  readonly outputs: Readonly<Record<string, boolean>>;
}

const ERROR_KEYS = new Set(["type", "message", "code"]);
const PROGRAM_KEYS = new Set(["name", "kind", "module", "version", "signature", "interface", "answer", "saved", "file", "line"]);
const SAW_KEYS = new Set(["call", "steps", "without", "slot", "saw_of"]);

/**
 * What a form maker keeps of an event of a kind it knows: the keys it knows,
 * and inside an `error`, a `program` and `saw`'s entries, the members it
 * knows. A `saw` entry it does not know becomes `{}`.
 */
export function known(e: EventLike): Rec {
  const keys = KEYS[e.kind as string] ?? [];
  const out: Rec = {};
  for (const [k, v] of Object.entries(e)) if (ENVELOPE.includes(k) || keys.includes(k)) out[k] = structuredClone(v);
  const pick = (o: unknown, allowed: Set<string>) =>
    Object.fromEntries(Object.entries(o as Rec).filter(([k]) => allowed.has(k)));
  if (out["error"] && typeof out["error"] === "object") out["error"] = pick(out["error"], ERROR_KEYS);
  if (out["program"] && typeof out["program"] === "object") out["program"] = pick(out["program"], PROGRAM_KEYS);
  if (Array.isArray(out["saw"])) {
    out["saw"] = (out["saw"] as unknown[]).map((x) =>
      typeof x === "object" && x !== null && !Array.isArray(x) && Object.keys(x).every((k) => SAW_KEYS.has(k)) ? x : {});
  }
  return out;
}

/**
 * The kept form of one event (streaming.md, "The kept form"), its `after`
 * not yet set; null when the kept form leaves it out. `keep` says which of
 * its call's fields are kept; `program` is its call's (`kind`, `answer`).
 */
export function keptEvent(e: EventLike, keep: KeptFields, program: { kind: string; answer: string }): Rec | null {
  const kind = e.kind as string;
  if (!KINDS.has(kind)) return null;
  const out = known(e);
  const all = { ...keep.inputs, ...keep.outputs };
  if (Object.values(all).every(Boolean)) return out;
  switch (kind) {
    case "started": {
      const given = (out["inputs"] ?? {}) as Rec;               // the copy: nothing a receiver does reaches the call's own event
      const inputs = Object.fromEntries(Object.entries(given).filter(([k]) => getOwn(keep.inputs, k) === true));
      const shaped: Rec = {};
      for (const [k, v] of Object.entries(out)) {
        if (k === "inputs") continue;
        shaped[k] = k === "content" ? false : v;
        if (k === "content") {
          shaped["omitted"] = {
            inputs: Object.keys(keep.inputs).filter((k) => !keep.inputs[k]),
            outputs: Object.keys(keep.outputs).filter((k) => !keep.outputs[k]),
          };
        }
      }
      if (Object.keys(inputs).length) shaped["inputs"] = inputs;
      return shaped;
    }
    case "request":
      return out;
    case "text":
      return getOwn(keep.outputs, e["field"] as string) === true ? out : null;
    case "thinking":
      return null;
    case "tool_call":
      delete out["input"];
      break;
    case "tool_result":
      delete out["output"];
      break;
    case "retry":
      delete out["reason"];
      break;
    case "done": {
      const holds = program.kind === "ai" ? [program.answer] : Object.keys(keep.outputs);
      if (holds.every((k) => getOwn(keep.outputs, k) === true)) return out;
      delete out["value"];
      break;
    }
    case "failed":
      out["error"] = Object.fromEntries(Object.entries(out["error"] as Rec).filter(([k]) => k === "type" || k === "code"));
      break;
  }
  out["content"] = false;
  return out;
}

/** A form of a log as it is made: each event's `after` is the position of the event before it in this form. */
export class Relink {
  last: Position | null;
  constructor(last: Position | null = null) {
    this.last = last;
  }
  take<E extends Rec>(e: E): E {
    const out = { ...e, after: this.last };
    this.last = positionOf(out);
    return out;
  }
}

/** The kept form of a whole log, given which fields each call keeps (by call id). */
export function keptLog(events: readonly EventLike[], kept: Readonly<Record<string, KeptFields>>): Rec[] {
  const programs = new Map<string, { kind: string; answer: string }>();
  for (const e of events) if (e.kind === "started") programs.set(e["call"] as string, e["program"] as { kind: string; answer: string });
  const link = new Relink();
  const out: Rec[] = [];
  for (const e of events) {
    const k = keptEvent(e, kept[e["call"] as string]!, programs.get(e["call"] as string) ?? { kind: "ai", answer: "result" });
    if (k) out.push(link.take(k));
  }
  return out;
}

// ------------------------------------------------------------------ a store

/** Why a store refuses an append, a claim or a read (streaming.md, "The rules a store keeps"). */
export type StoreCode = "event-malformed" | "event-conflict" | "event-gap" | "event-after-end" | "event-start" | "event-unknown";

/** A store's answer to an append (of one event, or a batch of one log's events). */
export type AppendAnswer = "kept" | "duplicate" | { readonly refuses: StoreCode; readonly event: Position };

/** A store's answer to a claim: the later writer's number and the last kept event, or a refusal. */
export type ClaimAnswer = { readonly writer: number; readonly after: Position } | { readonly refuses: "event-unknown" | "event-after-end" };

/**
 * Anything that keeps logs for others to read: a journal's store. Each claim
 * and each append is one step per log. A store that cannot take a claim and
 * an append as one step (a compare-and-set) gives no `claim`.
 */
export interface EventStore extends EventSource {
  /**
   * Append one log's events, in order, as one step: kept whole or not at
   * all. Answer `"kept"`, `"duplicate"` or a refusal. Throwing, rejecting
   * or not answering in time is no answer (the events are sent again); any
   * other answer is a refusal. The events are the store's own copies. `signal`
   * aborts when the journal stops waiting for this append (its `timeout`):
   * stop then if you can; a late answer is not read (the writer sends
   * again, and a kept event is then a duplicate).
   */
  append(events: readonly Readonly<Rec>[], opts?: { signal?: AbortSignal }): Promise<AppendAnswer>;
  read(tree: string, after: Position | null): Promise<ReadAnswer>;
  /** A later writer claims an unfinished log: the next writer number, fencing every earlier one. */
  claim?(tree: string): Promise<ClaimAnswer>;
}

const finishedLog = (log: readonly EventLike[], tree: string) => {
  const last = log[log.length - 1];
  return last !== undefined && last["call"] === tree && (last.kind === "done" || last.kind === "failed");
};

/**
 * A store in this process's memory, by the contract's rules: claims, appends
 * (single or batched, each one step), reads. For tests, and for a process
 * that keeps its own logs.
 */
export class MemoryStore implements EventStore {
  private logs = new Map<string, Rec[]>();
  private writers = new Map<string, number>();

  /** The events kept of a tree, in order. */
  events(tree: string): Rec[] {
    return structuredClone(this.logs.get(tree) ?? []);
  }

  /** The trees it holds. */
  trees(): string[] {
    return [...this.logs.keys()].filter((t) => this.logs.get(t)!.length);
  }

  /** Whether a tree's log is finished (its outermost call's end is kept). */
  finished(tree: string): boolean {
    return finishedLog(this.logs.get(tree) ?? [], tree);
  }

  /** The last writer number given for a tree (1 before any claim). */
  writerOf(tree: string): number {
    return this.writers.get(tree) ?? 1;
  }

  claimNow(tree: string): ClaimAnswer {
    const log = this.logs.get(tree) ?? [];
    if (!log.length) return { refuses: "event-unknown" };
    if (this.finished(tree)) return { refuses: "event-after-end" };
    const writer = this.writerOf(tree) + 1;
    this.writers.set(tree, writer);
    return { writer, after: positionOf(log[log.length - 1]!) };
  }

  private one(log: Rec[], writers: Map<string, number>, e: Rec): "kept" | "duplicate" | StoreCode {
    const tree = e["tree"] as string;
    const after = (e["after"] ?? null) as Position | null;
    if (!passes("event", e) || e["functai_event"] !== 2) return "event-malformed";
    if (after !== null && ((e["seq"] as number) <= after.seq || after.writer > (e["writer"] as number))) return "event-malformed";
    if (log.length && e["writer"] !== (writers.get(tree) ?? 1)) return "event-conflict";
    const had = log.find((k) => k["seq"] === e["seq"]);
    if (had) return lmcc.canonicalJson(had as lmcc.Json) === lmcc.canonicalJson(e as lmcc.Json) ? "duplicate" : "event-conflict";
    if (finishedLog(log, tree)) return "event-after-end";
    const last = log.length ? positionOf(log[log.length - 1]!) : null;
    if (!samePosition(after, last)) return (after?.seq ?? 0) > (last?.seq ?? 0) ? "event-gap" : "event-conflict";
    if (!log.length && (e.kind !== "started" || e["call"] !== tree || e["writer"] !== 1)) return "event-start";
    log.push(structuredClone(e));
    return "kept";
  }

  appendNow(events: readonly Readonly<Rec>[]): AppendAnswer {
    if (!events.length) return "duplicate";
    const tree = events[0]!["tree"] as string;
    const trial = [...(this.logs.get(tree) ?? [])];
    const answers: string[] = [];
    for (const e of events) {
      const answer = e["tree"] !== tree || typeof tree !== "string" ? "event-malformed" : this.one(trial, this.writers, e as Rec);
      if (answer !== "kept" && answer !== "duplicate") return { refuses: answer, event: positionOf(e) };
      answers.push(answer);
    }
    this.logs.set(tree, trial);
    return answers.every((a) => a === "duplicate") ? "duplicate" : "kept";
  }

  readNow(tree: string, after: Position | null): ReadAnswer {
    return resume(this.logs.get(tree) ?? [], after) as ReadAnswer;
  }

  async claim(tree: string): Promise<ClaimAnswer> {
    return this.claimNow(tree);
  }

  async append(events: readonly Readonly<Rec>[]): Promise<AppendAnswer> {
    return this.appendNow(events);
  }

  async read(tree: string, after: Position | null): Promise<ReadAnswer> {
    return structuredClone(this.readNow(tree, after));
  }
}

/**
 * What a journal says of an end its writer could not confirm
 * (streaming.md, "A required journal that does not confirm"): `"kept"` (the
 * log holds that event), `"another-end"` (it does not, and another writer
 * ended the log), or `"not-kept"` (it does not, and the log is unfinished:
 * final only once the caller has claimed the log).
 */
export async function settle(store: EventSource, tree: string, event: Position,
  opts: { signal?: AbortSignal } = {}): Promise<"kept" | "not-kept" | "another-end"> {
  const { signal } = opts;
  signal?.throwIfAborted();
  const reading = Promise.resolve(store.read(tree, null));
  let answer: ReadAnswer;
  if (!signal) answer = await reading;
  else {
    // a store that is likely unwell (it did not answer the end) may not answer this read either: the signal stops the wait
    let onAbort!: () => void;
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => reject(signal.reason);
      signal.addEventListener("abort", onAbort, { once: true });
    });
    reading.catch(() => undefined);
    try {
      answer = await Promise.race([reading, aborted]);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }
  const log = "events" in answer ? answer.events : [];
  if (log.some((e) => samePosition(positionOf(e), event))) return "kept";
  return finishedLog(log, tree) ? "another-end" : "not-kept";
}

// ------------------------------------------------------------------ receivers across layers

/** A journal as a layer sets it: the store, and whether calls wait for it. */
export interface JournalChoice<S = unknown> {
  readonly store: S;
  readonly mode: "required" | "best-effort";
}

/** One layer around a tree's outermost call, closest first: what it sets. */
export interface ReceiverLayer<O = unknown, S = unknown> {
  readonly where: "own" | "block" | "configure";
  readonly observers?: readonly O[];
  /** Absent: the layer sets none; null: it sets "no journal". */
  readonly journal?: JournalChoice<S> | null;
}

const sameJournal = (a: JournalChoice | null, b: JournalChoice | null) =>
  a === null || b === null ? a === b : a.store === b.store && a.mode === b.mode;

/** The indexes of the layers whose journal setting a farther layer refuses. */
function refusedSettings(layers: readonly ReceiverLayer[]): number[] {
  const setting = layers.map((l, i) => [i, l] as const).filter(([, l]) => "journal" in l && l.journal !== undefined)
    .map(([i, l]) => [i, l.journal ?? null] as const);
  const out = new Set<number>();
  setting.forEach(([i, far], n) => {
    for (const [k, near] of setting.slice(0, n)) {
      if (sameJournal(near, far)) continue;
      if (far !== null && far.mode === "required") out.add(k);          // replaces, weakens or removes a required journal
      else if (far !== null && layers[k]!.where === "own" && layers[i]!.where !== "own"
        && !(near !== null && near.store === far.store && near.mode === "required")) out.add(k);   // a program replaces or removes a host's
    }
  });
  return [...out].sort((a, b) => a - b);
}

/**
 * The observers and the journal a tree gets from the layers around its
 * outermost call, closest first (streaming.md, "Keeping a log while it is
 * written"): observers add up, outermost first; the closest journal setting
 * decides, except that a program's own setting cannot replace or remove a
 * host's journal, and no closer layer can replace, weaken or remove a
 * required one (`refused`: then `observers` and `journal` are where the
 * refused tree's own events go).
 */
export function receivers<O, S>(layers: readonly ReceiverLayer<O, S>[]): { observers: O[]; journal: JournalChoice<S> | null; refused: boolean } {
  const observers = [...layers].reverse().flatMap((l) => [...(l.observers ?? [])]);
  const refused = refusedSettings(layers);
  if (refused.length) {
    const rest = layers.slice(Math.max(...refused) + 1);
    return { observers, journal: receivers(rest).journal, refused: true };
  }
  const set = layers.find((l) => "journal" in l && l.journal !== undefined);
  return { observers, journal: set ? (set.journal ?? null) : null, refused: false };
}
