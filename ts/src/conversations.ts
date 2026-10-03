/**
 * Conversations: a program's calls that remember each other
 * (contract/conversations.md).
 *
 * ```ts
 * const chat = tutor.conversation("alex", { store: "tutoring/" });   // opened by id: the same line tomorrow reopens it
 * await chat("Hi, I'm Alex.");                                      // called like its program: one turn
 * const [first] = await chat.turns();
 * const other = chat.continueFrom(first);                           // a branch: nothing is ever deleted
 * chat.render("What is 1/2 + 1/3?");                                // the next request, nothing sent
 * ```
 *
 * A conversation is records in a store (stores.ts), appended and never
 * changed: a `turn` record before the call (the turn's id is its call's id,
 * known before the model is asked), `ended` after it, `lease` records while it
 * runs, and what resuming it needs (`reply`, `tool`, `waiting`, `approval`).
 * Memory belongs to the conversation, never to the program: `tutor` is
 * unchanged and still callable on its own.
 *
 * Inside a module's turn, helpers remember nothing unless the conversation
 * says so (`remembers: [[answer, "conversation"]]`), and `earlier()` is the
 * conversation so far, as data.
 */

import * as lmcc from "lmcc";
import { Response } from "@lm15/lm15";
import * as bridge from "lmcc/lm15";
import * as calllog from "./calllog.ts";
import { ACTIVE_TURN, type Call, type TurnRunLike } from "./calllog.ts";
import { Cancelled, ConversationError, Waiting, type Approval } from "./errors.ts";
import { positionOf, type Position } from "./events.ts";
import { Context as Slot } from "./host.ts";
import { dataShape, type Interface } from "./interface.ts";
import type { Observer } from "./log.ts";
import { aroundLayers, Applied, ContextHook, runHook, texts, TurnEndHook, TurnStartHook, type ConversationLike, type Plugin,
  type ShownTurn } from "./plugins.ts";
import { droppedFields, TurnWaiting } from "./program.ts";
import { checkSettings, layersOf, withSettings, type Settings } from "./settings.ts";
import { follow, isPersistent, storeOf, waitFor, type ConversationStore } from "./stores.ts";
import type { Stream } from "./stream.ts";
import { holderName } from "./replies.ts";
import { copyData, entriesOf, getOwn, recordOf, setOwn, toJson } from "./values.ts";
import { View } from "./views.ts";

type Rec = Record<string, unknown>;

/** The conversation records' format. */
export const FORMAT = 1;
/** Seconds a running turn's lease lasts. */
const LEASE = 30;
/** How often its process renews it (seconds). */
const RENEW = 10;
/** How often a running turn looks for a stop from another process (ms). */
const POLL = 250;

/** The clock a turn's lease is compared with (seconds); tests set it. */
export const clock = { now: () => Date.now() / 1000 };
const iso = (t?: number) => calllog.iso(t === undefined ? Date.now() : t * 1000);
const unix = (text: unknown): number => {
  if (typeof text !== "string" || !text) return 0;
  const t = Date.parse(text);
  return Number.isNaN(t) ? 0 : t / 1000 + (Number(/\.\d{3}(\d{3})Z$/.exec(text)?.[1] ?? 0) / 1e6);
};
const json = (v: unknown) => toJson(v)[0];
const record = (kind: string, fields: Rec): Rec => ({ functai_conversation: FORMAT, kind, at: iso(), ...fields });

// ------------------------------------------------------------------ what the model sees

/** Which earlier turns a turn is shown: every one (`last` null) or the last `last`; `without`: fields left out of every earlier turn. */
export interface ContextRule {
  readonly last: number | null;
  readonly without: readonly string[];
}

/**
 * Show the model every earlier turn (the default): running out of the
 * model's context, and being told, is better than a model that silently
 * misses what was said. `without`: inputs or outputs left out of every
 * earlier turn (a long document it already answered about).
 */
export function allTurns(opts: { without?: readonly string[] } = {}): ContextRule {
  return { last: null, without: [...(opts.without ?? [])] };
}

/**
 * Show the model only the last `n` earlier turns. The turns left out are
 * still kept, and each turn's record says which it was shown, so an answer
 * can be asked again exactly as it was.
 */
export function lastTurns(n: number, opts: { without?: readonly string[] } = {}): ContextRule {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`lastTurns takes a whole number of turns, not ${n}`);
  return { last: n, without: [...(opts.without ?? [])] };
}

const pick = <T>(rule: ContextRule, turns: readonly T[]): T[] =>
  rule.last === null ? [...turns] : rule.last > 0 ? turns.slice(-rule.last) : [];

/** What a helper remembers in a module's conversation. */
export interface Memory {
  readonly mode: "conversation" | "turn";
  readonly steps: boolean;
}

/**
 * What an AI function called inside a module's conversation remembers:
 * `"conversation"` (its own earlier calls on this branch, in earlier turns
 * and this one) or `"turn"` (its earlier calls in this turn only); `steps`:
 * with their tool calls and results. Helpers remember nothing otherwise.
 */
export function remember(mode: "conversation" | "turn" = "conversation", opts: { steps?: boolean } = {}): Memory {
  if (mode !== "conversation" && mode !== "turn") throw new TypeError(`a helper remembers "conversation" or "turn", not ${JSON.stringify(mode)}`);
  return { mode, steps: Boolean(opts.steps) };
}

const memoryOf = (v: unknown): Memory | "own" => {
  if (v === "own") return "own";
  if (v === "conversation" || v === "turn") return { mode: v, steps: false };
  if (typeof v === "object" && v !== null && "mode" in v) return v as Memory;
  throw new TypeError(`remembers maps a helper to "conversation", "turn", remember(...), or "own" (a conversation used inside); not ${JSON.stringify(v)}`);
};

// ------------------------------------------------------------------ the records, read

/** What the records say of one turn. */
export class TurnState {
  readonly record: Rec;
  ended: Rec | null = null;
  lease: Rec | null = null;
  waiting: Rec | null = null;
  stops = 0;
  readonly answers: Rec[] = [];
  readonly tools: Rec[] = [];
  readonly replies: Rec[] = [];
  readonly calls: Rec[] = [];
  readonly children: string[] = [];
  attempt = 1;
  constructor(record: Rec) {
    this.record = record;
  }

  get id(): string {
    return this.record["turn"] as string;
  }

  get parent(): string | null {
    return (this.record["parent"] ?? null) as string | null;
  }

  /** `running`, `waiting`, `interrupted` (its lease ran out with no end), or how it ended (`done`, `failed`, `stopped`, `abandoned`). */
  state(now = clock.now()): string {
    if (this.ended) return this.ended["state"] as string;
    const leaseSeq = (this.lease?.["seq"] ?? 0) as number;
    if (this.waiting && (this.waiting["seq"] as number) > leaseSeq) return "waiting";
    const until = this.lease ? unix(this.lease["until"]) : 0;
    return until >= now ? "running" : "interrupted";
  }

  /** The approvals the turn waits for that have no answer yet. */
  unanswered(): Rec[] {
    if (!this.waiting) return [];
    const done = new Set(this.answers.filter((a) => (a["seq"] as number) > (this.waiting!["seq"] as number)).map(akey));
    return ((this.waiting["approvals"] ?? []) as Rec[]).filter((a) => !done.has(akey(a)));
  }

  /** Tools that started and have no result in the records: they may have run. */
  unfinished(): Rec[] {
    const started = new Map<string, Rec>();
    for (const t of this.tools) {
      const k = `${t["site"]}|${t["invocation"]}`;
      if (t["state"] === "started") started.set(k, t);
      else if (t["state"] === "done" || t["state"] === "given" || t["state"] === "rerun") started.delete(k);
    }
    return [...started.values()];
  }
}

/** Which question an approval record answers: the asking call's site, the invocation, the plugin that asked. */
const akey = (a: Rec) => `${a["site"] ?? ""}|${Number(a["invocation"] ?? 0)}|${a["plugin"] ?? "approval"}`;

/** A conversation's records, applied in order (read again incrementally). */
export class ConvLog {
  n = 0;
  readonly turns = new Map<string, TurnState>();
  readonly order: string[] = [];
  head: string | null = null;
  readonly requestIds = new Map<string, string>();
  readonly programs = new Map<string, Rec>();
  readonly entries: Rec[] = [];

  apply(records: readonly Rec[]): this {
    for (const r of records) {
      this.n = Math.max(this.n, Number(r["seq"] ?? this.n + 1));
      if (r["functai_conversation"] !== FORMAT) continue;            // a format this reader does not know: skipped
      const kind = r["kind"];
      if (kind === "program") {
        if (!this.programs.has(r["version"] as string)) this.programs.set(r["version"] as string, r);
        continue;
      }
      if (kind === "entry") {
        this.entries.push(r);
        continue;
      }
      const tid = r["turn"] as string;
      if (kind === "turn") {
        if (this.turns.has(tid)) continue;
        this.turns.set(tid, new TurnState(r));
        this.order.push(tid);
        const parent = r["parent"] as string | null;
        if (parent && this.turns.has(parent)) this.turns.get(parent)!.children.push(tid);
        if (r["request_id"] && !this.requestIds.has(r["request_id"] as string)) this.requestIds.set(r["request_id"] as string, tid);
        this.head = tid;
        continue;
      }
      const st = this.turns.get(tid);
      if (kind === "head") {
        if (st) this.head = tid;
        continue;
      }
      if (!st) continue;
      if (kind === "ended") st.ended ??= r;
      else if (kind === "lease") {
        st.lease = r;
        st.attempt = Math.max(st.attempt, Number(r["attempt"] ?? 1));
      } else if (kind === "waiting") st.waiting = r;
      else if (kind === "stop") st.stops++;
      else if (kind === "approval") st.answers.push(r);
      else if (kind === "tool") st.tools.push(r);
      else if (kind === "reply") st.replies.push(r);
      else if (kind === "call") st.calls.push(r);
    }
    return this;
  }

  /** The turns from the first to `turn`, in order. */
  branch(turn: string | null): TurnState[] {
    const out: TurnState[] = [];
    const seen = new Set<string>();
    while (turn !== null && this.turns.has(turn) && !seen.has(turn)) {
      seen.add(turn);
      out.push(this.turns.get(turn)!);
      turn = this.turns.get(turn)!.parent;
    }
    return out.reverse();
  }

  /** `turn` when it ended `done`, else its nearest ancestor that did: a turn that did not end `done` is never a parent. */
  doneOn(turn: string | null): string | null {
    while (turn !== null && this.turns.has(turn)) {
      if (this.turns.get(turn)!.state() === "done") return turn;
      turn = this.turns.get(turn)!.parent;
    }
    return null;
  }
}

// ------------------------------------------------------------------ programs, as a conversation sees them

/** What a conversation needs of its program (an AI function or a module, made by `ai()` or `module()`). */
export interface ConversationProgram {
  readonly name: string;
  readonly module: string;
  readonly version: string;
  readonly interface: Interface;
  /** @internal */ readonly _kind: "ai" | "module";
  /** @internal The fields a turn's record shows: an AI function's signature's, a module's interface's. */
  _conversationFields(): Rec[];
  /** @internal The call's inputs, bound to the interface, as the record holds them (throws `InterfaceError`). */
  _recordedInputs(input: unknown): Rec;
  /** @internal A call, watched: `passive` streams nothing from the provider (read for its result only). */
  _stream(input: unknown, options: Rec, passive: boolean): Stream;
  /** @internal The program record's `program` facts (`program.signature` for AI functions). */
  _programInfo(): calllog.Program;
  /** @internal The AI functions a module calls (for `remembers`). */
  _aiFunctions?(): readonly object[];
  /** @internal A module's answer as it is written: the AI function whose answer text is the module's (views). */
  readonly _answerFrom?: string | null;
  readonly tools?: readonly unknown[];
  readonly signatureId?: string;
}

/** A program's descriptor record (conversations.md, `program`). */
function describe(program: ConversationProgram): Rec {
  const info = program._programInfo();
  const out: Rec = record("program", {
    version: info.version, name: info.name, program_kind: info.kind, module: info.module,
    interface: copyData(program.interface), fields: program._conversationFields(), answer: info.answer,
  });
  if (info.signature) out["signature"] = info.signature;
  return out;
}

/** A program's own fields as data for its descriptor: name, direction, purpose, shape without defaults. */
export function interfaceFields(iface: Interface): Rec[] {
  return [...iface.inputs.map((f) => ["input", f] as const), ...iface.outputs.map((f) => ["output", f] as const)]
    .map(([d, f]) => ({ name: f.name, direction: d, purpose: "plain", shape: dataShape(f.shape) }));
}

/**
 * Refuse a program that cannot be shown its earlier turns (decision 20): it
 * now writes an output they lack, unless `earlierWithout` names it and it is
 * one FunctAI adds (reasoning, tool calls); or a field changed its shape or
 * went away.
 */
function checkSignature(program: ConversationProgram, log: ConvLog, turns: readonly TurnState[], earlierWithout: readonly string[]): void {
  const now = new Map(program._conversationFields().map((f) => [f["name"] as string, f]));
  const seen = new Set<string>();
  for (const st of turns) {
    const version = st.record["program"] as string;
    if (seen.has(version) || !log.programs.has(version)) continue;
    seen.add(version);
    const was = new Map(((log.programs.get(version)!["fields"] ?? []) as Rec[]).map((f) => [f["name"] as string, f]));
    const same = (a: unknown, b: unknown) => lmcc.canonicalJson(a as lmcc.Json) === lmcc.canonicalJson(b as lmcc.Json);
    if (was.size === now.size && [...was].every(([n, f]) => now.has(n) && same(f, now.get(n)))) continue;
    const changed = [...was.keys()].filter((n) => now.has(n) && !same(was.get(n), now.get(n))).sort();
    const gone = [...was.keys()].filter((n) => !now.has(n)).sort();
    const added = [...now.keys()].filter((n) => !was.has(n)).sort();
    const hidden = (n: string) => ["reasoning", "tools.calls"].includes(now.get(n)!["purpose"] as string);
    const unexplained = added.filter((n) => !(hidden(n) && earlierWithout.includes(n)));
    if (changed.length || gone.length || unexplained.length) {
      const parts: string[] = [];
      if (unexplained.length) {
        const h = unexplained.filter(hidden);
        parts.push(`it now writes ${unexplained.join(", ")}, which earlier turns lack`
          + (h.length ? ` (to go on: earlierWithout: ${JSON.stringify(h)}; earlier turns are shown without them, nothing is rewritten)` : ""));
      }
      if (changed.length) parts.push(`${changed.join(", ")} changed type`);
      if (gone.length) parts.push(`${gone.join(", ")} is no longer one of its fields`);
      throw new ConversationError("conversation-signature",
        `${program.name}: its earlier turns in this conversation were made with other inputs or outputs: ${parts.join("; ")}`);
    }
  }
}

// ------------------------------------------------------------------ what a turn is shown

/** An earlier turn as data, for a call to show: an lmcc turn (with `signature` and `steps`) or `{ inputs, outputs }`. */
export type ShownData = Rec;

/** What a call is shown as earlier turns: the turns, their call ids, and a function that finishes the `saw` entries. */
export interface Shown {
  readonly turns: readonly ShownData[];
  readonly ids: readonly string[];
  readonly finish: ((entries: Rec[]) => Rec[]) | null;
}

/** A turn with some fields left out (and so shown without its steps), and which of its fields were. */
function without(turn: ShownData, names: readonly string[]): [ShownData, string[]] {
  const had = new Set([...Object.keys((turn["inputs"] ?? {}) as Rec), ...Object.keys((turn["outputs"] ?? {}) as Rec)]);
  const gone = [...had].filter((n) => names.includes(n)).sort();
  if (!gone.length) return [turn, []];
  const drop = (o: unknown) => recordOf(entriesOf((o ?? {}) as Rec).filter(([k]) => !gone.includes(k)));
  return [{ inputs: drop(turn["inputs"]), outputs: drop(turn["outputs"]) }, gone];
}

/** `[{saw_of: parent}, <parent's entry>]` when the entries are exactly what the parent saw, then the parent (calls.md, *Saw*). */
function compress(entries: Rec[], parent: string | null, sawOfTurn: (turn: string) => Rec[] | null): Rec[] {
  if (entries.length < 2 || parent === null || entries[entries.length - 1]!["call"] !== parent) return entries;
  const before = sawOfTurn(parent);
  if (before === null) return entries;
  const same = (a: unknown, b: unknown) => lmcc.canonicalJson(a as lmcc.Json) === lmcc.canonicalJson(b as lmcc.Json);
  return same(before, entries.slice(0, -1)) ? [{ saw_of: parent }, entries[entries.length - 1]!] : entries;
}

/** The context a turn is made with: what it is shown, and what is recorded. */
export interface TurnContext {
  readonly parent: string | null;
  readonly turns: readonly ShownData[];
  readonly ids: readonly string[];
  readonly finish: (entries: Rec[]) => Rec[];
  /** What `earlier()` gives: one row per earlier turn shown. */
  readonly rows: readonly Rec[];
  readonly sections: readonly string[];
  changes: Rec[];
  /** What a `context` hook made it be shown (recorded with the turn), or null. */
  readonly recorded: Rec | null;
  /** A module's turn: its `saw`. */
  readonly moduleSaw?: readonly Rec[];
}

// ------------------------------------------------------------------ the turn running in this process

/** The turn whose call is being prepared now (fn.ts asks what it is shown). */
const PREPARING = new Slot<Call | null>();
/** A rated row being asked again (evaluate, the optimizers), with what its call was shown. */
const REPLAY = new Slot<RowReplay | null>();
/** What `render` shows the next turn's call: the program, what it is shown, its sections. */
const RENDERING = new Slot<{ program: object; shown: Shown; sections: readonly string[] } | null>();

/** One turn being run by this process: its place, what its calls are shown, what resuming it replays, and the records it writes. */
class TurnRun implements TurnRunLike {
  readonly conv: Conversation;
  readonly turn: string;
  readonly attempt: number;
  readonly context: TurnContext;
  readonly self: object;
  readonly durable: boolean;
  readonly controller = new AbortController();
  rootTaken = false;
  root: Call | null = null;
  startSeq = 0;
  settings: Settings = {};
  readonly usage: Record<string, number> = {};
  model: string | null = null;
  readonly helperCalls: Rec[] = [];
  readonly later: { writer: number; after: Position | null; at: string; requests: number } | null;
  private readonly replies = new Map<string, Rec[]>();
  private readonly tools = new Map<string, Rec>();
  private readonly unfinished = new Map<string, Rec>();
  private readonly answers = new Map<string, [boolean, string | null, string | null, boolean]>();
  /** The store's own copy of the turn's kept log: its appends, in order. */
  private sinkChain: Promise<void> = Promise.resolve();
  private sinkBuffer: Rec[] = [];
  private sinkScheduled = false;
  private sinkLast: Position | null = null;
  private writes: Promise<unknown> = Promise.resolve();

  constructor(conv: Conversation, turn: string, opts: {
    attempt: number; context: TurnContext; replay?: { replies: Rec[]; tools: Rec[]; answers: Rec[]; waitingSeq: number };
    later?: { writer: number; after: Position | null; at: string; requests: number } | null;
  }) {
    this.conv = conv;
    this.turn = turn;
    this.attempt = opts.attempt;
    this.context = opts.context;
    this.self = conv.program;
    this.durable = conv.durable;
    this.later = opts.later ?? null;
    const r = opts.replay;
    for (const rec of r?.replies ?? []) {
      const q = this.replies.get(rec["key"] as string) ?? [];
      q.push(rec["response"] as Rec);
      this.replies.set(rec["key"] as string, q);
    }
    for (const t of r?.tools ?? []) {
      const k = `${t["site"]}|${Number(t["invocation"] ?? 0)}`;
      if (t["state"] === "done" || t["state"] === "given") {
        this.tools.set(k, t);
        this.unfinished.delete(k);
      } else if (t["state"] === "rerun") {
        this.tools.delete(k);
        this.unfinished.delete(k);
      } else if (t["state"] === "started" && !this.tools.has(k)) this.unfinished.set(k, t);
    }
    for (const a of r?.answers ?? []) {
      this.answers.set(akey(a), [a["verdict"] === "yes", (a["reason"] ?? null) as string | null, (a["by"] ?? null) as string | null,
        Number(a["seq"] ?? 0) > (r?.waitingSeq ?? 0)]);
    }
  }

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  /** The conversation the turn is in (plugins' entries). */
  get conversation(): Conversation {
    return this.conv;
  }

  /** Append records, in order with this turn's other writes; a store that fails is said once, and the turn goes on. */
  private append(records: Rec[]): Promise<unknown> {
    const p = this.writes.then(() => this.conv.append(records));
    this.writes = p.catch(() => undefined);
    return p;
  }

  /** Every record this run asked to append has been written (or failed). */
  async written(): Promise<void> {
    await this.writes;
  }

  // ----- calllog's side

  laterWriter(call: Call) {
    return call.id === this.turn ? this.later : null;
  }

  eventsSink(call: Call): Observer | null {
    const events = this.conv.store.events;
    if (!events || call.id !== this.turn) return null;
    const flush = () => {
      this.sinkScheduled = false;
      const batch = this.sinkBuffer;
      this.sinkBuffer = [];
      if (!batch.length) return;
      this.sinkChain = this.sinkChain.then(async () => {
        try {
          const answer = await events.append(batch);
          if (typeof answer === "object" && "refuses" in answer) throw new Error(answer.refuses);
          this.sinkLast = positionOf(batch[batch.length - 1]!);
        } catch (err) {
          calllog.warnOnce(`conversation-events:${(err as Error).message}`, `a turn's events could not be kept in its store (${(err as Error).message}); the turn goes on, and its records are kept`);
        }
      });
    };
    const keep = (e: unknown) => {
      this.sinkBuffer.push(e as Rec);
      if (!this.sinkScheduled) {
        this.sinkScheduled = true;
        setTimeout(flush, 0);
      }
    };
    Object.defineProperty(keep, "name", { value: "conversation store" });
    Object.defineProperty(keep, "passive", { value: true });     // a copy for other processes: it never streams a request
    return keep as Observer;
  }

  /** The store's copy holds every event of the turn's log made so far (at most `ms`). */
  async drained(ms = 5000): Promise<void> {
    const deadline = Date.now() + ms;
    const want = this.root?.node?.log.position ?? null;
    while (Date.now() < deadline) {
      await this.sinkChain;
      if (!want || (this.sinkLast && this.sinkLast.writer === want.writer && this.sinkLast.seq >= want.seq) || !this.conv.store.events) return;
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  started(call: Call): void {
    if (call.id !== this.turn) return;
    this.root = call;                                    // the turn's own call: its record says which conversation and turn it is
    call.conversation = { id: this.conv.id, turn: this.turn, parent: this.context.parent };
    call.changes.push(...copyData(this.context.changes));
  }

  ended(call: Call, failed: boolean): void {
    for (const [k, v] of Object.entries(call.usage)) this.usage[k] = (this.usage[k] ?? 0) + v;
    const last = [...call.exchanges].reverse().find((e) => e.response);
    if (last) this.model = last.model;
    if (call.id === this.turn || failed) return;
    // a helper the conversation remembers keeps its call for later
    const memory = this.conv.remembered(call);
    if (!memory || memory === "own") return;
    const pred = call.value as { turn?: { toJSON(): unknown } } | undefined;
    if (!pred?.turn) return;
    let rec: Rec;
    try {
      const program = call.program();
      rec = record("call", {
        turn: this.turn, attempt: this.attempt, call: call.id, site: call.path,
        program: { name: program.name, module: program.module, signature: program.signature ?? null },
        lmcc: json(pred.turn.toJSON()), saw: copyData(call.saw),
      });
    } catch {
      calllog.warnOnce(`remember:${call.node?.name}`, `${call.node?.name}'s call has a value with no JSON form: the conversation cannot remember it`);
      return;
    }
    this.helperCalls.push(rec);
    void this.append([rec]);
  }

  // ----- what calls are shown

  async helperContext(program: object & { name: string; module: string; signatureId?: string }, memory: Memory): Promise<[ShownData[], string[]]> {
    const found: Rec[] = [];
    if (memory.mode === "conversation") {
      const log = await this.conv.readLog();
      for (const st of log.branch(this.context.parent)) {
        if (st.state() !== "done") continue;
        const final = Number(st.ended?.["attempt"] ?? st.attempt);
        found.push(...st.calls.filter((c) => Number(c["attempt"] ?? 1) === final));
      }
    }
    found.push(...this.helperCalls);
    const mine = found.filter((c) => (c["program"] as Rec)["name"] === program.name && (c["program"] as Rec)["module"] === program.module);
    const turns: ShownData[] = [];
    const ids: string[] = [];
    for (const c of mine) {
      let t = copyData(c["lmcc"]) as Rec;
      if (!memory.steps) t["steps"] = [];
      if ((c["program"] as Rec)["signature"] !== program.signatureId) t = { inputs: t["inputs"] ?? {}, outputs: t["outputs"] ?? {} };
      turns.push(t);
      ids.push(c["call"] as string);
    }
    return [turns, ids];
  }

  // ----- resuming: what was recorded

  recordedReply(key: string): Response | null {
    const q = this.replies.get(key);
    if (q?.length) return Response.fromJSON(bridge.toLm15(q.shift()) as never);
    return null;
  }

  noteReply(key: string, response: Response): void {
    if (!this.durable) return;
    void this.append([record("reply", { turn: this.turn, attempt: this.attempt, key, response: json(Response.toJSON(response)) })]);
  }

  recordedTool(call: Call, approval: Approval): string | null {
    const k = `${call.path}|${approval.invocation}`;
    if (this.tools.has(k)) return (this.tools.get(k)!["output"] ?? null) as string | null;
    if (this.unfinished.has(k)) {
      throw new ConversationError("turn-unfinished", `${approval.path} started before the turn stopped, and whether it ran is not known: `
        + `resume({ results: { ${approval.invocation}: <what it returned> } }) or resume({ rerun: [${approval.invocation}] })`, { turn: this.turn });
    }
    return null;
  }

  async toolStarted(call: Call, approval: Approval): Promise<void> {
    await this.append([record("tool", {
      turn: this.turn, attempt: this.attempt, site: call.path, invocation: approval.invocation, id: approval.id, name: approval.name,
      input: json(approval.input), effects: approval.effects, state: "started",
    })]);
  }

  async toolDone(call: Call, approval: Approval, output: string): Promise<void> {
    await this.append([record("tool", {
      turn: this.turn, attempt: this.attempt, site: call.path, invocation: approval.invocation, id: approval.id, name: approval.name,
      state: "done", output,
    })]);
  }

  recordedApproval(call: Call, approval: Approval): [boolean, string | null, string | null, boolean] | null {
    return this.answers.get(`${call.path}|${approval.invocation}|${approval.plugin}`) ?? null;
  }

  async noteApproval(call: Call, approval: Approval, allowed: boolean, reason: string | null, by: string | null): Promise<void> {
    await this.append([record("approval", {
      turn: this.turn, site: call.path, invocation: approval.invocation, path: approval.path, plugin: approval.plugin,
      verdict: allowed ? "yes" : "no", by, reason,
    })]);
  }

  pause(call: Call, approval: Approval): never {
    throw new TurnWaiting(`${approval.path} waits for a person's answer`, { ...approval, site: call.path });
  }
}

// ------------------------------------------------------------------ what a call is shown (fn.ts and module.ts ask)

/**
 * What a call of `program` being prepared now (or rendered) is shown as
 * earlier turns, or null when it is shown nothing: the turn's own call its
 * conversation's turns, a remembered helper its own earlier calls, a row
 * asked again its earlier turns.
 */
export async function contextFor(program: object & { name: string; module: string; signatureId?: string }, call: Call | null): Promise<Shown | null> {
  const replay = REPLAY.get();
  if (replay && call) {
    const found = replay.contextFor(program, call);
    if (found) return found;
  }
  if (call) {
    const run = call.turnRun as TurnRun | null;
    if (!run) return null;
    if (call.id === run.turn) return { turns: run.context.turns, ids: run.context.ids, finish: run.context.finish };
    const memory = run.conv.rememberedProgram(program);
    if (memory && memory !== "own") {
      const [turns, ids] = await run.helperContext(program, memory);
      return { turns, ids, finish: null };
    }
    return null;
  }
  const rendering = RENDERING.get();
  if (rendering && rendering.program === program) return rendering.shown;
  return null;
}

/** The sections a call is given before its own `beforeCall` hooks: its turn's, a row's asked again, or, rendering, the next turn's. */
export function sectionsFor(program: object, call: Call | null): string[] {
  const replay = REPLAY.get();
  if (replay && call) return replay.sectionsFor(program, call);
  if (!call) {
    const rendering = RENDERING.get();
    return rendering && rendering.program === program ? [...rendering.sections] : [];
  }
  const run = call.turnRun as TurnRun | null;
  return run && call.id === run.turn ? [...run.context.sections] : [];
}

/** What a module's call is shown as the conversation so far (its `saw`), or null. */
export function moduleSaw(program: object, call: Call): Rec[] | null {
  const replay = REPLAY.get();
  if (replay && call.parent === null && replay.program === program) return replay.moduleSaw();
  const run = call.turnRun as TurnRun | null;
  if (run && call.id === run.turn) return [...(run.context.moduleSaw ?? [])];
  return null;
}

/**
 * The conversation so far, as data: inside a module's turn, one row per
 * earlier turn it is shown (its inputs and outputs by name); `[]` outside a
 * conversation. For a helper that declares an input for it.
 */
export function earlier(): Rec[] {
  const replay = REPLAY.get();
  if (replay) return replay.rows();
  let call = calllog.current.get() ?? null;
  const run = (call?.turnRun ?? ACTIVE_TURN.get() ?? null) as TurnRun | null;
  void call;
  call = null;
  return run ? copyData([...run.context.rows]) : [];
}

// ------------------------------------------------------------------ turns

/**
 * One turn of a conversation, as its records said when it was read: `id`
 * (also `call`: the id of its call in the call log), `parent`, `inputs`,
 * `outputs`, `result` (the answer), `state` (`running`, `waiting`,
 * `interrupted`, `done`, `failed`, `stopped`, `abandoned`), `model`,
 * `error`, `waiting` (the approvals it waits for), `unfinished` (tools that
 * may have run when it stopped), `usage`. `refresh()` reads it again.
 */
export class Turn {
  /** @internal */ readonly _conv: Conversation;
  /** @internal */ _st: TurnState;
  constructor(conv: Conversation, st: TurnState) {
    this._conv = conv;
    this._st = st;
  }

  /** The turn's id, which is also the id of its call in the call log (`call`). */
  get id(): string {
    return this._st.id;
  }

  get call(): string {
    return this._st.id;
  }

  /** The id of the conversation it belongs to. */
  get conversation(): string {
    return this._conv.id;
  }

  /** The id of the turn it continues (null for a conversation's first turn). */
  get parent(): string | null {
    return this._st.parent;
  }

  /** The `requestId` it was sent with: the same id sent again is this turn, not a new one. */
  get requestId(): string | null {
    return (this._st.record["request_id"] ?? null) as string | null;
  }

  /** Its inputs, by name, as they were recorded (a copy). */
  get inputs(): Rec {
    return copyData((this._st.record["inputs"] ?? {}) as Rec);
  }

  /** Every output by name (`result`, `reasoning`, …), once it is done; `{}` before. */
  get outputs(): Rec {
    return copyData((this._st.ended?.["outputs"] ?? {}) as Rec);
  }

  /** The answer (as the program's code returned it); null until done. */
  get result(): unknown {
    const ended = this._st.ended ?? {};
    if (Object.hasOwn(ended, "value")) return copyData(ended["value"]);
    return copyData(getOwn((ended["outputs"] ?? {}) as Rec, this._conv.answerName) ?? null);
  }

  /** Where it is: `running`, `waiting` (for a person's answer), `interrupted` (its process stopped), `done`, `failed`, `stopped`, `abandoned`. */
  get state(): string {
    return this._st.state();
  }

  /** The model that answered (or, before the end, the one it was asked to use). */
  get model(): string | null {
    return ((this._st.ended?.["model"] ?? (this._st.record["settings"] as Rec | undefined)?.["lm"]) ?? null) as string | null;
  }

  /** How a failed turn failed (`{ type, code, message }`); null otherwise. */
  get error(): Rec | null {
    return copyData((this._st.ended?.["error"] ?? null) as Rec | null);
  }

  /** Tokens, summed over every model call inside it. */
  get usage(): Record<string, number> {
    return { ...((this._st.ended?.["usage"] ?? {}) as Record<string, number>) };
  }

  /** A merge: the turns it was made from. */
  get reads(): string[] {
    return [...((this._st.record["reads"] ?? []) as string[])];
  }

  /** A merge: the program that made it (its rating goes there). */
  get madeBy(): Rec | null {
    return copyData((this._st.record["made_by"] ?? null) as Rec | null);
  }

  /** The tool calls it waits for a person to approve; `[]` when none. Answer with `approve()` or `deny()`. */
  get waiting(): Approval[] {
    return this._st.unanswered().map(approvalOf);
  }

  /**
   * Tools that started and have no result: the process stopped while they
   * ran, so they may have run. Before the turn goes on, say what each
   * returned (`resume({ results: { [invocation]: output } })`) or run it
   * again (`resume({ rerun: [invocation] })`): a tool is never run again on
   * its own.
   */
  get unfinished(): Rec[] {
    return this._st.unfinished().map((t) => ({
      invocation: t["invocation"], id: t["id"], name: t["name"], input: t["input"] ?? null, started: t["at"] ?? null, site: t["site"] ?? null,
    }));
  }

  /** The earlier turns this turn was shown, in order. */
  async saw(): Promise<Turn[]> {
    const log = await this._conv.readLog();
    const ids = expandSaw(log, this.id);
    return ids.filter((id) => log.turns.has(id)).map((id) => new Turn(this._conv, log.turns.get(id)!));
  }

  /** The turn as its records say now. */
  async refresh(): Promise<Turn> {
    const log = await this._conv.readLog();
    this._st = log.turns.get(this.id) ?? this._st;
    return this;
  }

  /** Wait until it is no longer running; resolves to it as it is then. */
  async wait(opts: { timeout?: number } = {}): Promise<Turn> {
    const deadline = opts.timeout === undefined ? Infinity : Date.now() + opts.timeout;
    for (;;) {
      await this.refresh();
      if (this._st.state() !== "running") return this;
      const left = deadline - Date.now();
      if (left <= 0) throw new Error(`turn ${this.id} is still running`);
      await waitFor(this._conv.store, this._conv.id, (await this._conv.readLog()).n, Math.min(500, left));
    }
  }

  /** Stop it wherever it runs: it ends `stopped` within a second. */
  async stop(): Promise<void> {
    await this._conv.stop(this);
  }

  /**
   * Say yes to an approval it waits for (`turn.waiting[0]`, its invocation
   * number, or nothing for the only one). When nothing else waits, the turn
   * goes on here (`resume: false`: later, `turn.resume()`); resolves to its answer.
   */
  approve(approval?: Approval | number | null, opts: { by?: string | null; resume?: boolean } = {}): Promise<unknown> {
    return this.answer(approval ?? null, true, null, opts.by ?? null, opts.resume ?? true);
  }

  /** Say no: the model is told the person did not allow it (and why). */
  deny(approval?: Approval | number | null, reason?: string | null, opts: { by?: string | null; resume?: boolean } = {}): Promise<unknown> {
    return this.answer(approval ?? null, false, reason ?? null, opts.by ?? null, opts.resume ?? true);
  }

  private async answer(approval: Approval | number | null, allowed: boolean, reason: string | null, by: string | null, resume: boolean): Promise<unknown> {
    await this.refresh();
    const waiting = this._st.unanswered();
    if (!waiting.length) throw new ConversationError("turn-state", `turn ${this.id} waits for no approval (it is ${this._st.state()})`, { turn: this.id });
    let target: Rec;
    if (approval === null) {
      if (waiting.length > 1) throw new TypeError(`turn ${this.id} waits for ${waiting.length} approvals: name one`);
      target = waiting[0]!;
    } else {
      const inv = typeof approval === "number" ? approval : approval.invocation;
      const site = typeof approval === "number" ? null : approval.site || null;
      const plugin = typeof approval === "number" ? null : approval.plugin || null;
      const match = waiting.filter((a) => Number(a["invocation"]) === Number(inv) && (site === null || a["site"] === site)
        && (plugin === null || (a["plugin"] ?? "approval") === plugin));
      if (!match.length) throw new ConversationError("turn-state", `turn ${this.id} waits for no approval ${inv}`, { turn: this.id });
      target = match[0]!;
    }
    await this._conv.append([record("approval", {
      turn: this.id, site: target["site"] ?? null, invocation: target["invocation"], path: target["path"] ?? null,
      plugin: target["plugin"] ?? "approval", verdict: allowed ? "yes" : "no", by, reason,
    })]);
    await this.refresh();
    if (resume && !this._st.unanswered().length) return this.resume();
    return undefined;
  }

  /**
   * Go on with a turn that waits (every approval answered) or was
   * interrupted (its process stopped), in this process: its program runs
   * again with the same inputs and earlier turns, each model reply and each
   * tool result it kept are reused, and it goes on from where it stopped.
   * Resolves to its answer (or rejects with `Waiting` again).
   */
  async resume(opts: { results?: Readonly<Record<string | number, unknown>>; rerun?: readonly (number | string)[] } = {}): Promise<unknown> {
    const { s } = await this._conv.resumeTurn(this.id, opts.results ?? {}, opts.rerun ?? []);
    return s.result;
  }

  /** End a turn that waits or was interrupted, without going on: `abandoned`. */
  async abandon(): Promise<void> {
    await this.refresh();
    const state = this._st.state();
    if (state !== "waiting" && state !== "interrupted") {
      throw new ConversationError("turn-state", `only a waiting or interrupted turn can be abandoned; turn ${this.id} is ${state}`, { turn: this.id });
    }
    await this._conv.append([record("ended", { turn: this.id, state: "abandoned", attempt: this._st.attempt })]);
    await this.refresh();
  }

  /**
   * Its events from its store (the kept form, or a view made from it:
   * `view: "outside"`), after the event named by `after`: those kept so far,
   * then each as it is kept, until its last. What another process, a page
   * after a reload, reads.
   */
  async *events(opts: { after?: Position | null; view?: "kept" | "outside"; timeout?: number; signal?: AbortSignal } = {}): AsyncGenerator<Rec> {
    const stop = async () => ["waiting", "interrupted", "abandoned"].includes((await this.refresh())._st.state());
    const follower = { stop, timeout: opts.timeout, signal: opts.signal };
    if ((opts.view ?? "kept") === "kept") {
      yield* follow(this._conv.store, this.id, opts.after ?? null, follower);
      return;
    }
    // a view is made from the kept form from its first event; a reader resumes it after a position it holds
    const v = new View(opts.view!, { answerFrom: this._conv.program._answerFrom ?? null });
    let waitingFor = opts.after ?? null;
    for await (const e of follow(this._conv.store, this.id, null, follower)) {
      const shown = v.apply(e);
      if (!shown) continue;
      if (waitingFor) {
        if (shown["writer"] === waitingFor.writer && shown["seq"] === waitingFor.seq) waitingFor = null;
        continue;
      }
      yield shown;
    }
    if (waitingFor) throw new ConversationError("event-unknown", `this view of turn ${this.id} has no event ${waitingFor.writer}-${waitingFor.seq}`);
  }

  /** The calls inside it, as its kept log says: `{ call, parent, function, invocation, ended }` each, in the order they started. */
  async calls(): Promise<Rec[]> {
    const events = this._conv.store.events;
    if (!events) return [];
    const got = await events.read(this.id, null);
    if ("refuses" in got) return [];
    const out = new Map<string, Rec>();
    for (const e of got.events as unknown as Rec[]) {
      if (e["kind"] === "started") {
        out.set(e["call"] as string, { call: e["call"], parent: e["parent"] ?? null, function: e["function"], invocation: e["invocation"] ?? null,
          ended: null, module: ((e["program"] ?? {}) as Rec)["module"] ?? null });
      } else if ((e["kind"] === "done" || e["kind"] === "failed") && out.has(e["call"] as string)) out.get(e["call"] as string)!["ended"] = e["kind"];
    }
    return [...out.values()];
  }

  /** The ids of the calls of `program` inside it (to rate one: `rate((await turn.find(fn))[0], "right")`). */
  async find(program: { name: string } | string): Promise<string[]> {
    const name = typeof program === "string" ? program : program.name;
    return (await this.calls()).filter((c) => c["function"] === name).map((c) => c["call"] as string);
  }

  /** The calls inside it, as an indented tree. */
  async tree(): Promise<string> {
    const calls = await this.calls();
    const kids = new Map<string | null, Rec[]>();
    for (const c of calls) {
      const k = c["call"] === this.id ? null : (c["parent"] as string | null);
      kids.set(k, [...(kids.get(k) ?? []), c]);
    }
    const lines: string[] = [];
    const walk = (c: Rec, prefix: string, last: boolean, top: boolean) => {
      const state = c["ended"] === "done" ? "" : ` [${c["ended"] ?? "running"}]`;
      lines.push(prefix + (top ? "" : last ? "└─ " : "├─ ") + c["function"] + state);
      const children = kids.get(c["call"] as string) ?? [];
      children.forEach((k, i) => walk(k, prefix + (top ? "" : last ? "   " : "│  "), i === children.length - 1, false));
    };
    for (const root of kids.get(null) ?? []) walk(root, "", true, true);
    return lines.join("\n");
  }

  toString(): string {
    const ins = Object.values(this.inputs).map((v) => short(v)).join(", ");
    return `Turn(${this.id.slice(0, 8)} ${ins}${this.state === "done" ? ` → ${short(this.result)}` : ` [${this.state}]`})`;
  }
}

function approvalOf(a: Rec): Approval {
  return {
    call: (a["call"] ?? "") as string, invocation: Number(a["invocation"]), id: (a["id"] ?? "") as string, name: (a["name"] ?? "") as string,
    input: copyData(a["input"] ?? null), effects: (a["effects"] ?? null) as Approval["effects"], path: (a["path"] ?? a["name"]) as string,
    site: (a["site"] ?? "") as string, plugin: (a["plugin"] ?? "approval") as string, question: (a["question"] ?? null) as string | null,
  };
}

const short = (v: unknown, n = 60): string => {
  const text = (typeof v === "string" ? v : JSON.stringify(v) ?? String(v)).split(/\s+/).join(" ");
  return text.length <= n ? text : text.slice(0, n - 1) + "…";
};

/** The turn ids a turn saw, `saw_of` expanded (an entry this reader does not know: nothing rather than a guess). */
function expandSaw(log: ConvLog, turn: string, following: readonly string[] = []): string[] {
  const st = log.turns.get(turn);
  const entries = ((st?.ended?.["saw"] ?? st?.waiting?.["saw"] ?? []) as Rec[]);
  const out: string[] = [];
  for (const [i, e] of entries.entries()) {
    if ("saw_of" in e) {
      const target = e["saw_of"] as string;
      if (i !== 0 || following.includes(target) || target === turn) return [];
      out.push(...expandSaw(log, target, [...following, turn]));
    } else if (typeof e["call"] === "string") out.push(e["call"]);
  }
  return out;
}

// ------------------------------------------------------------------ the conversation

/** The options of a conversation, besides settings for every turn. */
export interface ConversationOptions extends Settings {
  /** Where it is kept: null (this process's memory), a folder, true (the default folder), or any store with `append` and `read`. */
  store?: ConversationStore | string | boolean | null;
  /** Which earlier turns are shown: `allTurns()` (default) or `lastTurns(10)`, each with `without`. */
  context?: ContextRule;
  /** Outputs the program now writes that earlier turns lack (reasoning turned on, a first tool): earlier turns are shown without them. */
  earlierWithout?: readonly string[];
  /** A module's helpers' memory: `[[answer, "conversation"]]`, `"turn"`, `remember(...)`, or `"own"` for a conversation used inside it. */
  remembers?: Iterable<readonly [object, unknown]> | Map<object, unknown>;
  /** Two sends at once: `"queue"` (default: the second waits and continues from the first), `"refuse"` (`conversation-busy`), or `"branch"`. */
  sends?: "queue" | "refuse" | "branch";
}

/** One turn's options: settings for that turn only, a signal to stop it, and a request id (the same id sent twice is one turn). */
export interface TurnOptions extends Settings {
  signal?: AbortSignal;
  requestId?: string | null;
}

const RUNNING = new Map<string, [TurnRun, TurnStream]>();

/**
 * A program's conversation: its turns, kept in a store, called like the
 * program (`await chat(input, options)`). Made by `fn.conversation(...)` or
 * `module.conversation(...)`.
 */
export interface ConversationCall<I = unknown, A = unknown> {
  (input: I, options?: TurnOptions): Promise<A>;
}

export class Conversation {
  readonly program: ConversationProgram;
  readonly id: string;
  readonly store: ConversationStore;
  readonly context: ContextRule;
  readonly earlierWithout: readonly string[];
  readonly sends: "queue" | "refuse" | "branch";
  readonly settings: Settings;
  /** @internal */ readonly remembers: Map<object, Memory | "own">;
  /** @internal */ _head: string | null | typeof FOLLOW;
  /** @internal */ _exact = false;
  /** @internal */ _delegated = false;
  private log = new ConvLog();
  private sending: Promise<unknown> = Promise.resolve();

  constructor(program: ConversationProgram, id: string | null = null, opts: ConversationOptions = {}) {
    if (!program || (program._kind !== "ai" && program._kind !== "module")) throw new TypeError("a conversation is with an AI function or a module");
    const { store, context, earlierWithout, remembers, sends, ...settings } = opts;
    this.program = program;
    this.id = id === null ? calllog.newId() : checkIdOf(id);
    this.store = storeOf(store);
    this.context = context ?? allTurns();
    if (typeof this.context !== "object" || !("last" in this.context)) throw new TypeError("context is allTurns() or lastTurns(n)");
    this.earlierWithout = [...(earlierWithout ?? [])];
    this.sends = sends ?? "queue";
    if (!["queue", "refuse", "branch"].includes(this.sends)) throw new TypeError(`sends is "queue", "refuse" or "branch", not ${JSON.stringify(sends)}`);
    checkSettings(settings, "conversation");
    this.settings = settings;
    this.remembers = new Map();
    for (const [k, v] of remembers instanceof Map ? remembers.entries() : (remembers ?? [])) {
      if (program._kind === "ai") throw new TypeError("remembers is for a module's helpers; an AI function's conversation is its own memory");
      this.remembers.set(k, memoryOf(v));
    }
    this.checkRemembers();
    this._head = FOLLOW;
    this.checkContent();
    const opaque = [...program.interface.inputs, ...program.interface.outputs].filter((f) => f.opaque).map((f) => f.name);
    if (opaque.length) {
      throw new ConversationError("conversation-opaque", `${program.name}: ${opaque.join(", ")} may hold values with no JSON form, and a conversation keeps its turns as data. Give ${opaque.length > 1 ? "them" : "it"} a type`);
    }
  }

  /** @internal Whether its turns record what resuming needs (their replies): a module, or an AI function with tools. */
  get durable(): boolean {
    return this.program._kind === "module" || Boolean(this.program.tools?.length);
  }

  /** @internal The answer's name. */
  get answerName(): string {
    return this.program.interface.outputs[this.program.interface.outputs.length - 1]!.name;
  }

  private checkRemembers(): void {
    if (!this.remembers.size) return;
    const reachable = this.program._aiFunctions?.() ?? [];
    for (const [k, v] of this.remembers) {
      if (v === "own") continue;
      if (!reachable.includes(k)) {
        throw new TypeError(`remembers names ${(k as { name?: string }).name ?? String(k)}, which ${this.program.name} does not call (its AI functions: ${reachable.map((f) => (f as { name: string }).name).join(", ") || "none"})`);
      }
    }
  }

  private checkContent(): void {
    if (!isPersistent(this.store)) return;
    const fields = this.program._conversationFields();
    const outputs = fields.filter((f) => f["direction"] === "output");
    const dropped = droppedFields({
      inputs: fields.filter((f) => f["direction"] === "input").map((f) => f["name"] as string),
      outputs: outputs.map((f) => f["name"] as string),
      added: outputs.filter((f) => f["purpose"] !== "plain").map((f) => f["name"] as string),
    }, (this.program as unknown as { settings?: Settings }).settings ?? {}, this.settings);
    if (dropped.length) {
      throw new ConversationError("conversation-content", `${this.program.name}: a logContent setting keeps ${dropped.join(", ")} out of every record, and this store keeps a conversation's records. A conversation that must remember what it may not keep refuses rather than forgets: keep it in memory (store: null), or let the store keep those fields`);
    }
  }

  /** @internal How a remembered helper's calls are shown (by the program object a call was made by). */
  remembered(call: Call): Memory | "own" | null {
    const p = call.program();
    for (const [k, v] of this.remembers) {
      const f = k as { name?: string; module?: string };
      if (f.name === p.name && f.module === p.module) return v;
    }
    return null;
  }

  /** @internal */
  rememberedProgram(program: object): Memory | "own" | null {
    return this.remembers.get(program) ?? null;
  }

  // ----- records

  /** @internal The records, read again (only what was appended since). */
  async readLog(): Promise<ConvLog> {
    const got = await this.store.read(this.id, this.log.n);
    this.log.apply(got as Rec[]);
    return this.log;
  }

  /** @internal */
  async append(records: readonly Rec[], opts: { expect?: number } = {}): Promise<number> {
    return this.store.append(this.id, records, opts);
  }

  // ----- where it continues from

  private viewHead(log: ConvLog): string | null {
    if (this._head === FOLLOW) return log.head;
    const pinned = this._head;
    if (this._exact || pinned === null) return pinned;
    for (const tid of [...log.order].reverse()) {
      let t: string | null = tid;
      while (t !== null) {
        if (t === pinned) return tid;
        t = log.turns.get(t)?.parent ?? null;
      }
    }
    return pinned;
  }

  /** The turn this conversation continues from next (null: none yet). */
  async head(): Promise<Turn | null> {
    const log = await this.readLog();
    const h = this.viewHead(log);
    return h !== null && log.turns.has(h) ? new Turn(this, log.turns.get(h)!) : null;
  }

  /** Make a turn the conversation's head, for everyone who opens it. */
  async setHead(turn: Turn | string | number): Promise<void> {
    const tid = await this.turnId(turn);
    await this.append([record("head", { turn: tid })]);
    this._head = tid;
    this._exact = true;
  }

  /** The turns from the first to the head, in order. */
  async turns(): Promise<Turn[]> {
    const log = await this.readLog();
    return log.branch(this.viewHead(log)).map((st) => new Turn(this, st));
  }

  /** Every turn of every branch, in the order they were made. */
  async allTurns(): Promise<Turn[]> {
    const log = await this.readLog();
    return log.order.map((t) => new Turn(this, log.turns.get(t)!));
  }

  /** One turn, by its id (or a Turn, or its index in `turns()`). */
  async turn(turn: Turn | string | number): Promise<Turn> {
    const tid = await this.turnId(turn);
    return new Turn(this, (await this.readLog()).turns.get(tid)!);
  }

  private async turnId(turn: Turn | string | number): Promise<string> {
    if (turn instanceof Turn) return turn.id;
    if (typeof turn === "number") {
      const all = await this.turns();
      const t = all.at(turn);
      if (t) return t.id;
    }
    if (typeof turn === "string" && (await this.readLog()).turns.has(turn)) return turn;
    throw new ConversationError("turn-unknown", `conversation ${this.id} has no turn ${JSON.stringify(turn)}`);
  }

  /**
   * This conversation, continuing after `turn` (a Turn or its id): the next
   * turn is a new branch. Nothing is deleted.
   */
  continueFrom(turn: Turn | string): this {
    const tid = turn instanceof Turn ? turn.id : turn;
    const view = Object.create(Conversation.prototype) as Conversation;
    Object.assign(view, this);
    view._head = tid;
    view._exact = true;
    view.log = new ConvLog();
    view.sending = Promise.resolve();
    return chatOf(view) as unknown as this;
  }

  // ----- plugin entries

  /**
   * Keep a plugin's entry in this conversation, at a turn (it then belongs to
   * the branches through that turn): what the plugin needs later, never shown
   * to the model by itself.
   */
  async remember(plugin: string, kind: string, data: unknown, opts: { turn?: string | null } = {}): Promise<void> {
    if (typeof kind !== "string" || !kind) throw new TypeError("an entry's kind is a name");
    await this.append([record("entry", { plugin, entry: kind, data: json(data), ...(opts.turn ? { turn: opts.turn } : {}) })]);
    await this.readLog();
  }

  /** A plugin's entries of a kind on the branch through `branch` (default: this view's head), oldest first: `{ turn, data, at }`. */
  entries(plugin: string, kind: string, opts: { branch?: string | null } = {}): Rec[] {
    const log = this.log;
    const through = opts.branch !== undefined ? opts.branch : this.viewHead(log);
    const on = new Set(log.branch(through).map((st) => st.id));
    return log.entries.filter((e) => e["plugin"] === plugin && e["entry"] === kind && (e["turn"] === undefined || e["turn"] === null || on.has(e["turn"] as string)))
      .map((e) => ({ turn: e["turn"] ?? null, data: copyData(e["data"]), at: e["at"] ?? null }));
  }

  // ----- calling it

  /** @internal The conversation, as plugins' events see it. */
  asHost(): ConversationLike {
    return {
      id: this.id,
      entries: (plugin, kind, opts) => this.entries(plugin, kind, opts),
      remember: async (plugin, kind, data, opts) => {
        await this.remember(plugin, kind, data, opts);
        await this.readLog();
      },
      branchOf: (turn) => this.log.branch(turn).map((st) => st.id),
      turnsOf: (parent) => this.log.branch(parent).filter((st) => st.state() === "done").map((st) => ({
        id: st.id, inputs: copyData((st.record["inputs"] ?? {}) as Rec), outputs: copyData((st.ended?.["outputs"] ?? {}) as Rec), without: [],
      })),
    };
  }

  private plugins(settings: Settings): Plugin[] {
    const own = (this.program as unknown as { settings?: Settings }).settings ?? {};
    const layers = layersOf(own);
    return aroundLayers([layers[0]!, { where: "block", settings }, ...layers.slice(1)]);
  }

  /** One turn: the program's answer. (The conversation is also callable: `await chat(input)`.) */
  call(input: unknown, options: TurnOptions = {}): Promise<unknown> {
    return this.send(input, options, true).result;
  }

  /** One turn of an AI function: the whole call (every output, the lmcc turn; `p.callId` is the turn's id). */
  predict(input: unknown, options: TurnOptions = {}): Promise<unknown> {
    return this.send(input, options, true).prediction;
  }

  /** One turn, watched while it is made: the turn is saved before the model is asked (`await s.turn`). */
  stream(input: unknown, options: TurnOptions = {}): TurnStream {
    return this.send(input, options, false);
  }

  /** @internal */
  send(input: unknown, options: TurnOptions, passive: boolean): TurnStream {
    const { signal, requestId, ...settings } = options;
    checkSettings(settings, "a turn");
    const s = new TurnStream(this, passive, signal);
    // two sends from this object are recorded one after the other (each reads what the one before wrote)
    const begun = this.sending.then(() => this.begin(input, settings, requestId ?? null));
    this.sending = begun.catch(() => undefined);
    s.begin(begun);
    return s;
  }

  private checkNested(): void {
    const call = calllog.current.get();
    const outer = (call?.turnRun ?? ACTIVE_TURN.get() ?? null) as TurnRun | null;
    if (!outer || this._delegated || (outer.conv.id === this.id && outer.conv.store === this.store)) return;
    for (const [k, v] of outer.conv.remembers) if ((k === this || k === this.program) && v === "own") return;
    throw new ConversationError("conversation-nested", `conversation ${this.id} (${this.program.name}) is used inside a turn of conversation ${outer.conv.id}, which does not say so: a remembering program inside another is refused unless declared (remembers: [[${this.program.name}, "own"]])`);
  }

  /** Record the turn (and its first lease), then start its call. */
  private async begin(input: unknown, settings: Settings, requestId: string | null): Promise<Begun> {
    this.checkNested();
    this.checkContent();                               // a host's rule set since the conversation was opened holds too
    const turnSettings: Settings = { ...this.settings, ...settings };
    const plugins = this.plugins(turnSettings);
    let [inputs, startChanges] = await this.turnStart(input, plugins);
    const values = this.program._recordedInputs(inputs);    // refused before anything is recorded
    inputs = values;
    for (;;) {
      const log = await this.readLog();
      if (requestId !== null && log.requestIds.has(requestId)) return { existing: log.requestIds.get(requestId)! };
      const parent = this.parentFor(log);
      if (parent !== BUSY) {
        checkSignature(this.program, log, log.branch(parent), this.earlierWithout);
        const tid = calllog.newId();
        const context = await this.contextOf(log, parent, turnSettings, null, plugins);
        context.changes = [...startChanges, ...context.changes];
        const desc = describe(this.program);
        const recs: Rec[] = [];
        if (!log.programs.has(desc["version"] as string)) recs.push(desc);
        const rec = record("turn", { turn: tid, parent, program: desc["version"], inputs: copyData(values) });
        if (requestId !== null) rec["request_id"] = String(requestId);
        if (typeof turnSettings.lm === "string") rec["settings"] = { lm: turnSettings.lm };
        if (context.recorded) rec["context"] = context.recorded;
        if (context.changes.length) rec["changes"] = context.changes;
        recs.push(rec, this.lease(tid, 1));
        let startSeq: number;
        try {
          startSeq = await this.append(recs, { expect: log.n });
        } catch (err) {
          if (err instanceof ConversationError && err.code === "store-conflict") continue;     // someone appended meanwhile: read again
          throw err;
        }
        this._head = tid;                            // from now on, this view follows its own branch
        this._exact = false;
        const run = new TurnRun(this, tid, { attempt: 1, context });
        run.settings = turnSettings;
        run.startSeq = startSeq;
        return { run, inputs, settings: turnSettings };
      }
      await waitFor(this.store, this.id, log.n, 250);      // a turn before it is running: queue behind it
    }
  }

  /** The `turnStart` hooks: the turn's inputs as they leave them. */
  private async turnStart(input: unknown, plugins: readonly Plugin[]): Promise<[unknown, Rec[]]> {
    if (!plugins.some((p) => p.handlers.turnStart?.length)) return [input, []];
    const names = this.program.interface.inputs.map((f) => f.name);
    const given = this.program._recordedInputs(input);
    const log = await this.readLog();
    const event = new TurnStartHook(copyData(given), this.id, this.viewHead(log), this.program);
    const applied = new Applied();
    await runHook("turnStart", plugins, event, applied, (c) => {
      const unknown = Object.keys(c.inputs ?? {}).filter((k) => !names.includes(k));
      if (unknown.length) throw new TypeError(`${this.program.name} has no input ${JSON.stringify(unknown[0])}`);
      event.inputs = { ...event.inputs, ...(c.inputs ?? {}) };
    });
    return [event.inputs, applied.items];
  }

  /** @internal The `turnEnd` hooks (they hear; they may keep entries at the turn), run before the turn's end is recorded. */
  async turnEnd(run: TurnRun, outcome: Rec): Promise<void> {
    const plugins = this.plugins({ ...this.settings, ...run.settings });
    if (!plugins.some((p) => p.handlers.turnEnd?.length)) return;
    const log = await this.readLog();
    const st = log.turns.get(run.turn)!;
    const event = new TurnEndHook(run.turn, outcome["state"] as string, copyData((st.record["inputs"] ?? {}) as Rec),
      copyData((outcome["outputs"] ?? {}) as Rec), this.asHost(), st.parent);
    for (const plugin of plugins) {
      for (const fn of plugin.handlers.turnEnd ?? []) {
        event._plugin = plugin;
        try {
          await fn(event);
        } catch (err) {
          calllog.warnOnce(`turn_end:${plugin.name}:${(err as Error)?.name}`, `plugin ${plugin.name} failed in turn_end (${(err as Error)?.name}: ${(err as Error)?.message})`);
        } finally {
          event._plugin = null;
        }
      }
    }
  }

  /** The turn a new turn continues from, or BUSY while it must wait. */
  private parentFor(log: ConvLog): string | null | typeof BUSY {
    const head = this.viewHead(log);
    if (head === null) return null;
    const st = log.turns.get(head);
    if (!st) return null;
    const state = st.state();
    if (state === "waiting" && this.sends !== "branch") {
      throw new ConversationError("conversation-busy", `conversation ${this.id}: turn ${head} waits for a person's answer (answer it, or continue from another turn)`, { turn: head });
    }
    if (state === "running" || state === "waiting") {
      if (this.sends === "queue") return BUSY;
      if (this.sends === "refuse") throw new ConversationError("conversation-busy", `conversation ${this.id}: turn ${head} is ${state}`, { turn: head });
      return log.doneOn(st.parent);
    }
    return log.doneOn(head);
  }

  /** @internal */
  lease(tid: string, attempt: number): Rec {
    return record("lease", { turn: tid, holder: holderName(), until: iso(clock.now() + LEASE), attempt });
  }

  /**
   * The earlier turns shown, fields left out of each, sections, changes made,
   * and the context to record: the conversation's rule picks among the done
   * turns of the branch, then the `context` hooks change it. `fixed`: the
   * context a turn recorded when it was made (resuming it shows exactly that).
   */
  private async shown(log: ConvLog, parent: string | null, plugins: readonly Plugin[], fixed: Rec | null):
    Promise<[TurnState[], Map<string, string[]>, string[], Rec[], Rec | null]> {
    const done = log.branch(parent).filter((st) => st.state() === "done");
    if (fixed) {
      const by = new Map(done.map((st) => [st.id, st]));
      const picked = ((fixed["turns"] ?? []) as string[]).filter((t) => by.has(t)).map((t) => by.get(t)!);
      const w = new Map(Object.entries((fixed["without"] ?? {}) as Record<string, string[]>).map(([k, v]) => [k, [...v]]));
      return [picked, w, [...((fixed["sections"] ?? []) as string[])], [], null];
    }
    let picked = pick(this.context, done);
    const w = new Map<string, string[]>();
    for (const st of picked) {
      const fields = new Set([...Object.keys((st.record["inputs"] ?? {}) as Rec), ...Object.keys((st.ended?.["outputs"] ?? {}) as Rec)]);
      const gone = [...fields].filter((f) => this.context.without.includes(f)).sort();
      if (gone.length) w.set(st.id, gone);
    }
    if (!plugins.some((p) => p.handlers.context?.length)) return [picked, w, [], [], null];
    const shownTurn = (st: TurnState): ShownTurn => ({ id: st.id, inputs: copyData((st.record["inputs"] ?? {}) as Rec),
      outputs: copyData((st.ended?.["outputs"] ?? {}) as Rec), without: [...(w.get(st.id) ?? [])] });
    const event = new ContextHook(picked.map(shownTurn), [], this.asHost(), parent, this.program);
    const applied = new Applied();
    const byDone = new Map(done.map((st) => [st.id, st]));
    await runHook("context", plugins, event, applied, (c) => {
      if (c.keep !== undefined) {
        const ids = [...c.keep];
        const unknown = ids.filter((i) => !byDone.has(i));
        if (unknown.length) throw new TypeError(`keep names ${unknown[0]}, which is not a done turn of this branch`);
        picked = done.filter((st) => ids.includes(st.id));
        event.turns = picked.map(shownTurn);
      }
      if (c.without !== undefined) {
        const targets: Record<string, readonly string[]> = Array.isArray(c.without)
          ? Object.fromEntries(picked.map((st) => [st.id, c.without as readonly string[]]))
          : c.without as Record<string, readonly string[]>;
        for (const [tid, names] of Object.entries(targets)) {
          if (typeof names === "string" || !Array.isArray(names) || !names.every((n) => typeof n === "string")) {
            throw new TypeError("without names fields: a list of names, or { [turn id]: names }");
          }
          w.set(tid, [...new Set([...(w.get(tid) ?? []), ...names])].sort());
        }
      }
      if (c.sections !== undefined) event.sections.push(...texts(c.sections));
    });
    const ids = new Set(picked.map((st) => st.id));
    for (const k of [...w.keys()]) if (!ids.has(k)) w.delete(k);
    const recorded = { turns: picked.map((st) => st.id), without: Object.fromEntries(w), sections: [...event.sections] };
    return [picked, w, [...event.sections], applied.items, recorded];
  }

  /**
   * What the turn after `parent` is shown: the earlier turns (each as the
   * lmcc turn it was, with its steps), the fields left out taken out; the
   * sections; the `saw` entries; the rows `earlier()` gives; the changes
   * plugins made, and the context to record with the turn.
   */
  /** @internal */
  async contextOf(log: ConvLog, parent: string | null, settings: Settings, fixed: Rec | null = null, plugins?: readonly Plugin[]): Promise<TurnContext> {
    const [picked, w, sections, changes, recorded] = await this.shown(log, parent, plugins ?? this.plugins(settings), fixed);
    const turns: ShownData[] = [];
    const ids: string[] = [];
    const rows: Rec[] = [];
    const dropped = new Map<string, string[]>();
    const isAi = this.program._kind === "ai";
    const signature = isAi ? this.program.signatureId : undefined;
    for (const st of picked) {
      const outputs = (st.ended?.["outputs"] ?? {}) as Rec;
      const row = { ...((st.record["inputs"] ?? {}) as Rec), ...outputs };
      const gone = w.get(st.id) ?? [];
      rows.push(recordOf(entriesOf(row).filter(([k]) => !gone.includes(k))));
      ids.push(st.id);
      if (!isAi) continue;
      const desc = log.programs.get(st.record["program"] as string) ?? {};
      let t: ShownData = st.ended && Object.hasOwn(st.ended, "lmcc") && desc["signature"] === signature
        ? copyData(st.ended["lmcc"] as Rec) : { inputs: st.record["inputs"] ?? {}, outputs };
      let left: string[];
      [t, left] = without(t, gone);
      if (left.length) dropped.set(st.id, left);
      turns.push(t);
    }
    const sawOfTurn = (turn: string): Rec[] | null => {
      try {
        const st = log.turns.get(turn);
        if (!st) return null;
        return expandEntries(log, turn, []);
      } catch {
        return null;
      }
    };
    const finish = (entries: Rec[]): Rec[] => {
      const out = entries.map((e) => {
        const gone = dropped.get(e["call"] as string);
        if (!gone) return e;
        const copy: Rec = { ...e, without: [...new Set([...((e["without"] ?? []) as string[]), ...gone])].sort() };
        delete copy["steps"];
        return copy;
      });
      return compress(out, parent, sawOfTurn);
    };
    const base = { parent, rows, sections, changes, recorded };
    if (!isAi) {
      const entries = finish(ids.map((i) => ({ call: i })));
      return { ...base, turns: [], ids: [], finish: () => entries, moduleSaw: entries };
    }
    return { ...base, turns, ids, finish };
  }

  /** @internal Start a turn that was recorded (or resumed): its call, under its settings, as this turn. */
  async start(run: TurnRun, inputs: unknown, settings: Settings, s: TurnStream): Promise<{ s: Stream }> {
    return { s: ACTIVE_TURN.run(run, () => withSettings(settings, (): Stream => this.program._stream(inputs, { signal: run.signal }, s.passive))) };
  }

  /** @internal */
  async resumeTurn(tid: string, results: Readonly<Record<string | number, unknown>>, rerun: readonly (number | string)[]): Promise<{ s: TurnStream }> {
    let attempt = 0;
    let startSeq = 0;
    for (;;) {
      const log = await this.readLog();
      const st = log.turns.get(tid);
      if (!st) throw new ConversationError("turn-unknown", `conversation ${this.id} has no turn ${tid}`);
      const state = st.state();
      if (state !== "waiting" && state !== "interrupted") throw new ConversationError("turn-state", `turn ${tid} is ${state}: only a waiting or interrupted turn goes on`, { turn: tid });
      if (st.unanswered().length) throw new ConversationError("turn-state", `turn ${tid} still waits for ${st.unanswered().length} approval(s)`, { turn: tid });
      const given: Rec[] = [];
      for (const t of st.unfinished()) {
        const inv = Number(t["invocation"]);
        const keys = [inv, String(inv), t["id"]];
        const hit = Object.entries(results).find(([k]) => keys.includes(k) || keys.includes(Number(k)));
        const base = { turn: tid, site: t["site"] ?? null, invocation: t["invocation"], id: t["id"] ?? null, name: t["name"] ?? null };
        if (hit) given.push(record("tool", { ...base, state: "given", output: typeof hit[1] === "string" ? hit[1] : json(hit[1]) }));
        else if (rerun.some((r) => keys.includes(r) || keys.includes(Number(r)))) given.push(record("tool", { ...base, state: "rerun" }));
        else throw new ConversationError("turn-unfinished", `turn ${tid}: ${t["name"]} (invocation ${inv}) started and may have run: resume({ results: { ${inv}: <what it returned> } }) or resume({ rerun: [${inv}] })`, { turn: tid });
      }
      attempt = st.attempt + 1;
      try {
        startSeq = await this.append([...given, this.lease(tid, attempt)], { expect: log.n });
      } catch (err) {
        if (err instanceof ConversationError && err.code === "store-conflict") continue;
        throw err;
      }
      break;
    }
    const log = await this.readLog();
    const st = log.turns.get(tid)!;
    let later: { writer: number; after: Position | null; at: string; requests: number } | null = null;
    const events = this.store.events;
    if (events?.claim) {
      try {
        const claim = await events.claim(tid);
        if (!("refuses" in claim)) {
          const kept = await events.read(tid, null);
          const list = "events" in kept ? kept.events as unknown as Rec[] : [];
          const requests = Math.max(0, ...list.filter((e) => e["kind"] === "request" && e["call"] === tid).map((e) => Number(e["request"] ?? 0)));
          later = { writer: claim.writer, after: claim.after, at: (list[list.length - 1]?.["at"] ?? "") as string, requests };
        }
      } catch {
        later = null;                                   // no log to continue: this writer starts none
      }
    }
    const fixed = (st.record["context"] ?? null) as Rec | null;
    const context = await this.contextOf(log, st.parent, this.settings, fixed);
    context.changes = [];                               // the turn's own changes are on its first record
    const run = new TurnRun(this, tid, {
      attempt, context, later,
      replay: { replies: st.replies, tools: st.tools, answers: st.answers, waitingSeq: Number(st.waiting?.["seq"] ?? 0) },
    });
    run.startSeq = startSeq;
    const settings: Settings = { ...this.settings };
    const lm = (st.record["settings"] as Rec | undefined)?.["lm"];
    if (typeof lm === "string") settings.lm = lm;
    run.settings = settings;
    const s = new TurnStream(this, true, undefined);
    s.begin(Promise.resolve({ run, inputs: copyData(st.record["inputs"] ?? {}), settings }));
    return { s };
  }

  /**
   * Stop a running turn, wherever it runs: in this process or another that
   * opened the same store. It ends `stopped` within about a second. A turn
   * that waits or was interrupted has nothing running it: stopping it ends it
   * `abandoned`. A turn that already ended is left as it is.
   */
  async stop(turn: Turn | string): Promise<void> {
    const tid = await this.turnId(turn);
    const t = await this.turn(tid);
    if (t.state === "waiting" || t.state === "interrupted") return t.abandon();
    if (t.state !== "running") return;
    await this.append([record("stop", { turn: tid })]);
    RUNNING.get(tid)?.[1].close();
  }

  /**
   * The exact request the next turn would send (nothing is sent or recorded).
   * `call: helper`: the request that helper would get, with the memory the
   * conversation gives it (the inputs are then the helper's).
   */
  async render(input: unknown, opts: { call?: object & { render(i: unknown, s?: Settings): unknown; name: string; module: string; signatureId?: string } } & Settings = {}): Promise<unknown> {
    const { call: helper, ...settings } = opts;
    const log = await this.readLog();
    let parent = this.viewHead(log);
    parent = parent !== null ? log.doneOn(parent) : null;
    if (helper) {
      const memory = this.remembers.get(helper);
      let shown: Shown = { turns: [], ids: [], finish: null };
      if (memory && memory !== "own") {
        const run = new TurnRun(this, calllog.newId(), { attempt: 1, context: { parent, turns: [], ids: [], finish: (e) => e, rows: [], sections: [], changes: [], recorded: null } });
        const [turns, ids] = await run.helperContext(helper, memory);
        shown = { turns, ids, finish: null };
      }
      return RENDERING.run({ program: helper, shown, sections: [] }, () => helper.render(input, settings));
    }
    if (this.program._kind !== "ai") throw new TypeError(`${this.program.name} is a module: render the request one of its helpers would get, chat.render(input, { call: helper })`);
    const ctx = await this.contextOf(log, parent, this.settings);
    const program = this.program as unknown as { render(i: unknown, s?: Settings): unknown };
    return RENDERING.run({ program: this.program, shown: { turns: ctx.turns, ids: ctx.ids, finish: null }, sections: ctx.sections },
      () => withSettings(this.settings, () => program.render(input, settings)));
  }

  /**
   * A turn after this conversation's head made from several branches by
   * another AI function (`fn`): its answer becomes this program's answer,
   * recorded as a turn (`madeBy` fn, `reads` the branches), so the next turn
   * sees it. `inputs`: `fn`'s; when `fn` has one input and none is given, it
   * is given the branches' answers (`[{ model, ...inputs, ...outputs }]`).
   */
  async merge(branches: readonly (Turn | string)[], fn: { predict(i: unknown): Promise<{ answer: unknown; callId: string }>; interface: Interface; name: string; version: string },
    inputs?: Rec): Promise<Turn> {
    if (this.program._kind !== "ai") throw new TypeError("merge makes an AI function's turn; a module's conversation cannot take one");
    if (!branches.length) throw new TypeError("merge needs the turns to merge");
    const log = await this.readLog();
    const read = await Promise.all(branches.map((b) => this.turn(b)));
    const parent = this.viewHead(log);
    for (const t of read) {
      if (t.parent !== parent) throw new ConversationError("turn-state", `turn ${t.id} does not continue from ${parent}: a merge reads branches of the turn it follows`, { turn: t.id });
      if (t.state !== "done") throw new ConversationError("turn-state", `turn ${t.id} is ${t.state}`, { turn: t.id });
    }
    let given = inputs;
    if (!given) {
      const names = fn.interface.inputs.map((f) => f.name);
      if (names.length !== 1) throw new TypeError(`${fn.name} takes ${names.length} inputs: give them by name`);
      given = { [names[0]!]: read.map((t) => ({ model: t.model, ...t.inputs, ...t.outputs })) };
    }
    const p = await fn.predict(given);
    const answer = this.answerName;
    const program = this.program as unknown as { _exampleTurn(inputs: Rec, outputs: Rec): Rec };
    let lmccJson: Rec;
    try {
      lmccJson = program._exampleTurn(read[0]!.inputs, { [answer]: p.answer });
    } catch (err) {
      throw new TypeError(`${fn.name}'s answer does not fit ${this.program.name}'s answer: ${(err as Error).message}`);
    }
    const tid = calllog.newId();
    const desc = describe(this.program);
    const recs: Rec[] = log.programs.has(desc["version"] as string) ? [] : [desc];
    recs.push(record("turn", { turn: tid, parent, program: desc["version"], inputs: read[0]!.inputs, reads: read.map((t) => t.id),
      made_by: { name: fn.name, version: fn.version, call: p.callId } }));
    recs.push(record("ended", { turn: tid, state: "done", outputs: { [answer]: json(p.answer) }, value: json(p.answer), lmcc: lmccJson, saw: [], model: null, attempt: 1 }));
    await this.append(recs);
    this._head = tid;
    this._exact = false;
    return this.turn(tid);
  }

  toString(): string {
    return `Conversation(${this.id} with ${this.program.name})`;
  }
}

/** A conversation that is also callable like its program: `await chat(input, options)` is one turn. */
export function chatOf<C extends Conversation>(conv: C): C & ((input: unknown, options?: TurnOptions) => Promise<unknown>) {
  const f = function chat(input: unknown, options?: TurnOptions) {
    return (f as unknown as Conversation).call(input, options);
  };
  Object.setPrototypeOf(f, Conversation.prototype);
  Object.assign(f, conv);
  return f as unknown as C & ((input: unknown, options?: TurnOptions) => Promise<unknown>);
}

const FOLLOW = Symbol("follow its head");
const BUSY = Symbol("busy");

function checkIdOf(id: string): string {
  if (typeof id !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/.test(id)) {
    throw new ConversationError("conversation-id", `a conversation's id is 1 to 200 letters, digits, '.', '_' or '-', starting with a letter or digit; not ${JSON.stringify(id)}`);
  }
  return id;
}

/** The entries a turn saw, `saw_of` expanded (calls.md, *Saw*), as the turn records hold them. */
function expandEntries(log: ConvLog, turn: string, following: readonly string[]): Rec[] {
  const st = log.turns.get(turn);
  const entries = (st?.ended?.["saw"] ?? st?.waiting?.["saw"]) as Rec[] | undefined;
  if (!entries) throw new Error("not recorded");
  const out: Rec[] = [];
  entries.forEach((e, i) => {
    if ("saw_of" in e) {
      const target = e["saw_of"] as string;
      if (i !== 0 || target === turn || following.includes(target)) throw new Error("not known");
      out.push(...expandEntries(log, target, [...following, turn]));
    } else out.push(copyData(e));
  });
  return out;
}

type Begun = { run: TurnRun; inputs: unknown; settings: Settings } | { existing: string };

// ------------------------------------------------------------------ a turn, watched

/**
 * A turn, watched while it is made: a stream (its answer's text as it is
 * written, its events, its result) with `turn` (known once it is recorded,
 * before the model is asked), and `approve`/`deny` for tool calls it waits
 * for in this process. When its call ends, the turn's `ended` (or `waiting`)
 * record is written before its result is given.
 */
export class TurnStream implements AsyncIterable<string>, PromiseLike<unknown> {
  readonly conversation: Conversation;
  /** Read for its result only: its replies are not streamed from the provider. */
  readonly passive: boolean;
  /** The turn, once recorded (before the model is asked). */
  readonly turn: Promise<Turn>;
  /** The answer (the same as calling the program); rejects with its error, `Waiting` when it waits for a person. */
  readonly result: Promise<unknown>;
  /** An AI function's turn: every output, the lmcc turn, the replies (`p.callId` is the turn's id). */
  readonly prediction: Promise<unknown>;
  /** @internal The program's own stream, boxed (a stream is thenable: a promise would adopt its answer). */
  readonly inner: Promise<{ s: Stream | null }>;
  private resolveInner!: (s: Stream | null) => void;
  private resolveTurn!: (t: Turn) => void;
  private rejectTurn!: (e: unknown) => void;
  private resolveResult!: (v: unknown) => void;
  private rejectResult!: (e: unknown) => void;
  private resolvePrediction!: (v: unknown) => void;
  private rejectPrediction!: (e: unknown) => void;
  private readonly controller = new AbortController();
  private turnId: string | null = null;
  private innerNow: Stream | null = null;

  constructor(conversation: Conversation, passive: boolean, signal: AbortSignal | undefined) {
    this.conversation = conversation;
    this.passive = passive;
    this.inner = new Promise((r) => { this.resolveInner = (s) => { this.innerNow = s; r({ s }); }; });
    this.turn = new Promise((ok, fail) => { this.resolveTurn = ok; this.rejectTurn = fail; });
    this.result = new Promise((ok, fail) => { this.resolveResult = ok; this.rejectResult = fail; });
    this.prediction = new Promise((ok, fail) => { this.resolvePrediction = ok; this.rejectPrediction = fail; });
    for (const p of [this.turn, this.result, this.prediction]) p.catch(() => undefined);
    if (signal) {
      if (signal.aborted) this.controller.abort();
      else signal.addEventListener("abort", () => this.close(), { once: true });
    }
  }

  /** @internal */
  begin(begun: Promise<Begun>): void {
    void this.run(begun);
  }

  private async run(begun: Promise<Begun>): Promise<void> {
    let b: Begun;
    try {
      b = await begun;
    } catch (err) {
      this.resolveInner(null);
      for (const f of [this.rejectTurn, this.rejectResult, this.rejectPrediction]) f(err);
      return;
    }
    const conv = this.conversation;
    if ("existing" in b) {
      // the same request id twice is one turn: this one watches the first (in this process, or through its store)
      this.turnId = b.existing;
      const here = RUNNING.get(b.existing);
      if (here) {
        this.resolveInner((await here[1].inner).s);
        here[1].turn.then(this.resolveTurn, this.rejectTurn);
        here[1].result.then(this.resolveResult, this.rejectResult);
        here[1].prediction.then(this.resolvePrediction, this.rejectPrediction);
        return;
      }
      this.resolveInner(null);
      try {
        const t = await conv.turn(b.existing);
        this.resolveTurn(t);
        const done = await t.wait();
        const value = outcome(conv, done);
        this.resolveResult(value);
        this.resolvePrediction({ outputs: done.outputs, answer: value, callId: done.id });
      } catch (err) {
        this.rejectResult(err);
        this.rejectPrediction(err);
      }
      return;
    }
    const { run } = b;
    this.turnId = run.turn;
    RUNNING.set(run.turn, [run, this]);
    const started = performance.now();
    if (this.controller.signal.aborted) run.controller.abort();
    else this.controller.signal.addEventListener("abort", () => run.controller.abort(), { once: true });
    const beat = this.heartbeat(run);
    let inner: Stream | null = null;
    try {
      inner = (await conv.start(run, b.inputs, b.settings, this)).s;
    } catch (err) {
      this.resolveInner(null);
      await this.conclude(run, undefined, err, started);
      beat.stop();
      return;
    }
    this.resolveInner(inner);
    void conv.turn(run.turn).then(this.resolveTurn, this.rejectTurn);
    let value: unknown;
    let error: unknown;
    let failed = false;
    let pred: unknown;
    try {
      value = await inner!.result;
      const p = (inner as unknown as { prediction?: Promise<unknown> }).prediction;
      pred = p ? await p : undefined;
    } catch (err) {
      failed = true;
      error = err;
    }
    beat.stop();
    await this.conclude(run, failed ? undefined : { value, pred }, failed ? error : undefined, started);
  }

  /** Renew the lease, look for a stop from another process, and stop when another process took the turn over. */
  private heartbeat(run: TurnRun): { stop(): void } {
    const conv = this.conversation;
    let seen = run.startSeq;
    let renewed = Date.now();
    let stopped = false;
    let busy = false;
    const tick = async () => {
      if (stopped || busy) return;
      busy = true;
      try {
        const recs = (await conv.store.read(conv.id, seen)) as Rec[];
        seen += recs.length;
        const mine = recs.filter((r) => r["turn"] === run.turn);
        if (mine.some((r) => r["kind"] === "stop")) this.close();
        if (mine.some((r) => r["kind"] === "lease" && Number(r["attempt"] ?? 1) > run.attempt)) {
          calllog.warnOnce(`lease-lost:${run.turn}`, `turn ${run.turn} was taken over by another process (its lease ran out here): this one stops`);
          this.close();
        }
        if (Date.now() - renewed >= RENEW * 1000) {
          renewed = Date.now();
          await conv.append([conv.lease(run.turn, run.attempt)]);
        }
      } catch (err) {
        calllog.warnOnce(`heartbeat:${(err as Error).name}`, `a turn's lease could not be renewed (${(err as Error).message})`);
      } finally {
        busy = false;
      }
    };
    const timer = setInterval(() => void tick(), POLL);
    (timer as { unref?: () => void }).unref?.();
    return { stop: () => { stopped = true; clearInterval(timer); } };
  }

  /** Write the turn's last record, then give the reader its outcome. */
  private async conclude(run: TurnRun, ok: { value: unknown; pred: unknown } | undefined, error: unknown, started: number): Promise<void> {
    const conv = this.conversation;
    try {
      await run.written();
      await run.drained();
      const root = run.root;
      const saw = root ? copyData(root.saw) : [...(run.context.moduleSaw ?? [])];
      const base = { turn: run.turn, attempt: run.attempt };
      if (error instanceof TurnWaiting) {
        const a = error.approval as Approval;
        await conv.append([record("waiting", { ...base, approvals: [approvalJson(a)], saw })]);
        const t = await conv.turn(run.turn);
        const err = new Waiting(`turn ${run.turn} waits for a person's answer: ${a.path}(${short(a.input, 80)})`, t, t.waiting);
        this.fail(err);
        return;
      }
      const rec: Rec = record("ended", { ...base, saw, seconds: Math.round((performance.now() - started) * 1000) / 1e6, usage: { ...run.usage } });
      let out: unknown = ok?.value;
      if (!ok) {
        rec["state"] = error instanceof Cancelled ? "stopped" : "failed";
        rec["error"] = calllog.errorJson(error);
      } else {
        rec["state"] = "done";
        try {
          const [outputs, lmccJson, model] = this.outputsOf(run, ok.value, ok.pred);
          rec["outputs"] = outputs;
          rec["value"] = json(ok.value);
          if (lmccJson !== null) rec["lmcc"] = lmccJson;
          rec["model"] = model;
        } catch (err) {
          rec["state"] = "failed";
          rec["error"] = { type: (err as Error).name, message: (err as Error).message };
          error = err;
          out = undefined;
        }
      }
      await conv.turnEnd(run, rec);                   // before the end is recorded: the next turn sees what it keeps
      try {
        await conv.append([rec]);
      } catch (err) {
        calllog.warnOnce(`ended:${(err as Error).name}`, `a turn's end could not be kept in its store (${(err as Error).message})`);
      }
      if (rec["state"] === "done") {
        this.resolveResult(out);
        this.resolvePrediction(ok!.pred ?? { outputs: rec["outputs"], answer: out, callId: run.turn });
      } else this.fail(error);
    } finally {
      RUNNING.delete(run.turn);
    }
  }

  private fail(err: unknown): void {
    this.rejectResult(err);
    this.rejectPrediction(err);
  }

  private outputsOf(run: TurnRun, value: unknown, pred: unknown): [Rec, Rec | null, string | null] {
    if (this.conversation.program._kind === "ai") {
      const p = pred as { outputs: Rec; turn: { toJSON(): unknown }; response: { model?: string } | null; toolCalls?: unknown } | undefined;
      if (!p?.turn) throw new TypeError("the turn's call left no prediction");
      const outputs = recordOf(entriesOf(p.outputs).map(([k, v]) => [k, json(v)] as const));
      const fields = this.conversation.program._conversationFields().filter((f) => f["direction"] === "output").map((f) => f["name"]);
      if (p.toolCalls !== undefined && fields.includes("calls")) outputs["calls"] = json(p.toolCalls);
      return [outputs, json(p.turn.toJSON()) as Rec, p.response?.model ?? run.model];
    }
    const iface = this.conversation.program.interface;
    const names = iface.outputs.map((f) => f.name);
    const named: Rec = names.length === 1 ? { [names[0]!]: value } : (value as Rec);
    return [recordOf(names.filter((n) => getOwn(named, n) !== undefined).map((n) => [n, json(getOwn(named, n))] as const)), null, run.model];
  }

  /** Stop the turn: no new request starts, and it ends `stopped`. */
  close(): void {
    this.controller.abort();
    void this.inner.then(({ s }) => s?.close());
  }

  /** The answer so far (its text since its latest request). */
  get text(): string {
    return this.innerNow?.text ?? "";
  }

  /** Every event of the turn's call tree, as it happens (see `Stream.events`). */
  async *events(opts: Parameters<Stream["events"]>[0] = {}): AsyncGenerator<Rec> {
    const { s } = await this.inner;
    if (s) {
      yield* s.events(opts) as unknown as AsyncGenerator<Rec>;
      return;
    }
    if (this.turnId) {
      const t = await this.turn.catch(() => null);
      if (t) yield* t.events({ view: opts.view ? "outside" : "kept" });
    }
  }

  /** The answer's text, piece by piece. */
  async *[Symbol.asyncIterator](): AsyncGenerator<string> {
    const { s } = await this.inner;
    if (s) {
      for await (const piece of s) yield piece;
    } else {
      for await (const e of this.events()) if (e["kind"] === "text" && e["answer"] && e["call"] === this.turnId) yield e["text"] as string;
    }
    await this.result;
  }

  /** Say yes to a tool call this turn waits for here (on a stream with no conversation store to wait in). */
  async approve(approval?: Approval | number | null, opts: { by?: string | null } = {}): Promise<void> {
    const { s } = await this.inner;
    (s as unknown as { approve(a?: unknown, o?: unknown): void } | null)?.approve(approval, opts);
  }

  async deny(approval?: Approval | number | null, reason?: string | null, opts: { by?: string | null } = {}): Promise<void> {
    const { s } = await this.inner;
    (s as unknown as { deny(a?: unknown, r?: unknown, o?: unknown): void } | null)?.deny(approval, reason, opts);
  }

  then<T1 = unknown, T2 = never>(ok?: ((value: unknown) => T1 | PromiseLike<T1>) | null, fail?: ((reason: unknown) => T2 | PromiseLike<T2>) | null): Promise<T1 | T2> {
    return this.result.then(ok, fail);
  }
}

function approvalJson(a: Approval): Rec {
  return {
    call: a.call, invocation: a.invocation, id: a.id, name: a.name, input: json(a.input), effects: a.effects, path: a.path,
    site: a.site, plugin: a.plugin, ...(a.question ? { question: a.question } : {}),
  };
}

function outcome(conv: Conversation, t: Turn): unknown {
  const state = t.state;
  if (state === "done") return t.result;
  if (state === "waiting") throw new Waiting(`turn ${t.id} waits for a person's answer`, t, t.waiting);
  const err = t.error ?? {};
  throw new ConversationError("turn-state", `turn ${t.id} ended ${state}${t.error ? `: ${err["type"]}: ${err["message"] ?? ""}` : ""}`, { turn: t.id });
}

// ------------------------------------------------------------------ rows asked again with their context (stage 5)

/**
 * A rated row asked again (`evaluate`, the optimizers) with what its call
 * was shown: the program's own call gets the row's `earlier` turns, each
 * helper call the earlier turns its original call was shown (in the order
 * they were made), and its record says `saw_of` the original. No
 * conversation is read or written.
 */
class RowReplay {
  readonly program: object;
  private readonly earlierTurns: Rec[];
  private readonly helpers: Rec[];
  private readonly sections: string[];
  private readonly call: string | null;
  private readonly helperSections = new Map<string, string[]>();

  constructor(row: Rec, program: object) {
    this.program = program;
    this.earlierTurns = [...((getOwn(row, "earlier") ?? []) as Rec[])];
    this.helpers = [...((getOwn(row, "helpers") ?? []) as Rec[])];
    this.sections = [...((getOwn(row, "sections") ?? []) as string[])];
    this.call = (calllog.metaOf(row, "call") ?? null) as string | null;
  }

  private turnOf(program: { signatureId?: string }, t: Rec): Rec {
    if (t["steps"] && t["signature"] === program.signatureId) {
      return { signature: t["signature"], inputs: copyData(t["inputs"] ?? {}), steps: copyData(t["steps"]), outputs: copyData(t["outputs"] ?? {}) };
    }
    return { inputs: copyData(t["inputs"] ?? {}), outputs: copyData(t["outputs"] ?? {}) };
  }

  contextFor(program: object & { name: string; signatureId?: string }, call: Call): Shown | null {
    if (call.parent === null && program === this.program) {
      if (!this.earlierTurns.length) return null;
      const original = this.call;
      return { turns: this.earlierTurns.map((t) => this.turnOf(program, t)), ids: this.earlierTurns.map(() => ""),
        finish: (e) => (original ? [{ saw_of: original }] : e) };
    }
    const i = this.helpers.findIndex((h) => h["program"] === program.name);
    if (i < 0) return null;
    const [h] = this.helpers.splice(i, 1);
    this.helperSections.set(call.id, [...((h!["sections"] ?? []) as string[])]);
    const turns = ((h!["earlier"] ?? []) as Rec[]).map((t) => this.turnOf(program, t));
    if (!turns.length) return null;
    const original = (h!["call"] ?? null) as string | null;
    return { turns, ids: turns.map(() => ""), finish: (e) => (original ? [{ saw_of: original }] : e) };
  }

  moduleSaw(): Rec[] {
    return this.call && this.earlierTurns.length ? [{ saw_of: this.call }] : [];
  }

  sectionsFor(program: object, call: Call): string[] {
    if (call.parent === null && program === this.program) return [...this.sections];
    return [...(this.helperSections.get(call.id) ?? [])];
  }

  rows(): Rec[] {
    return this.earlierTurns.map((t) => ({ ...((t["inputs"] ?? {}) as Rec), ...((t["outputs"] ?? {}) as Rec) }));
  }
}

/** Ask a rated row again with the context its call had (a row with none changes nothing). */
export function replaying<R>(row: Rec, program: object, fn: () => R): R {
  const has = (k: string) => {
    const v = getOwn(row, k);
    return Array.isArray(v) && v.length > 0;
  };
  if (!has("earlier") && !has("helpers") && !has("sections")) return fn();
  return REPLAY.run(new RowReplay(row, program), fn);
}

/** Whether a row was answered after earlier turns (it is measured, never made a worked example). */
export function inContext(row: Rec): boolean {
  const e = getOwn(row, "earlier");
  const h = getOwn(row, "helpers");
  return (Array.isArray(e) && e.length > 0) || (Array.isArray(h) && h.length > 0);
}

/** @internal Marks the call whose saw is being prepared. */
export { PREPARING };

/** @internal A conversation's state machine, for the contract's cases. */
export const internals = { ConvLog, TurnState, checkSignature, describe };
