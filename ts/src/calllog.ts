/**
 * The call log (contract/calls.md): every call of an AI function as one line
 * of JSON in a folder, ratings of those calls, and the rows with known
 * answers they make. The folder is the interface: Python, TypeScript and
 * any tool read and write the same one.
 *
 * Writing needs a file system (Node, Deno, Bun); in a browser calls are not
 * logged, and `rated` reads records you pass it.
 */

import * as lmcc from "lmcc";
import { Request, Response } from "@lm15/lm15";
import * as lm15 from "@lm15/lm15";

const LMCC_VERSION: unknown = (lmcc as unknown as { VERSION?: unknown }).VERSION;
const LM15_VERSION: unknown = (lm15 as unknown as { VERSION?: unknown }).VERSION;
import { writtenRecord, type CallFields } from "./content.ts";
import type { Node, Observer } from "./log.ts";
import { builtin, Context, env, pid, runtime } from "./host.ts";
import type { Settings } from "./settings.ts";
import { copyData, entriesOf, getOwn, parseData, recordOf, setOwn, toJson, writeData } from "./values.ts";
import { FunctAIError, type Approval } from "./errors.ts";
import { earlierOf, needsContext, recordsById, SawUnknown } from "./saw.ts";

/** The call record's format (calls.md): 2 since 2026-09-28. A reader reads 1 and 2. */
export const FORMAT = 2;
/** The formats of call records this reader reads. */
const CALL_FORMATS = new Set([1, 2]);
/** The rating record's format. */
export const RATING_FORMAT = 1;
const MAX_LINE = 8 * 1024 * 1024;
const OFF = new Set(["", "0", "false", "no", "off"]);
const ON = new Set(["1", "true", "yes", "on"]);
export const VERSION = "0.1.0";

type Json = unknown;
type Rec = Record<string, unknown>;

// ------------------------------------------------------------------ ids and times

/** A UUIDv7 (RFC 9562): time-ordered. */
export function newId(): string {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  let ms = Date.now();
  for (let i = 5; i >= 0; i--) {
    bytes[i] = ms % 256;
    ms = Math.floor(ms / 256);
  }
  bytes[6] = (bytes[6]! & 0x0f) | 0x70;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/** RFC 3339 UTC with exactly six fraction digits (milliseconds here, then 000). */
export function iso(ms: number): string {
  const d = new Date(Math.floor(ms));
  const whole = d.toISOString().slice(0, 19);
  const micro = Math.round((ms - Math.floor(ms / 1000) * 1000) * 1000);
  return `${whole}.${String(Math.min(micro, 999999)).padStart(6, "0")}Z`;
}

const now = () => (globalThis.performance ? globalThis.performance.timeOrigin + globalThis.performance.now() : Date.now());

// ------------------------------------------------------------------ values

export { toJson } from "./values.ts";

// ------------------------------------------------------------------ where

export function defaultFolder(): string | null {
  const os = builtin("node:os");
  const path = builtin("node:path");
  if (!os || !path) return null;
  const e = env();
  let base: string;
  if (os.platform() === "darwin") base = path.join(os.homedir(), "Library", "Application Support");
  else if (os.platform() === "win32") base = e["LOCALAPPDATA"] ?? path.join(os.homedir(), "AppData", "Local");
  else base = e["XDG_DATA_HOME"] || path.join(os.homedir(), ".local", "share");
  return path.join(base, "functai", "calls");
}

function expand(p: string): string {
  const os = builtin("node:os");
  const path = builtin("node:path");
  if (!path) return p;
  const home = p === "~" || p.startsWith("~/") ? (os?.homedir() ?? "~") + p.slice(1) : p;
  return path.resolve(home);
}

/** The folder calls are logged to under this setting, or null when off. */
export function folderOf(setting: Settings["logCalls"]): string | null {
  if (setting === false) return null;
  const raw = (env()["FUNCTAI_LOG_CALLS"] ?? "").trim();
  const on = !OFF.has(raw.toLowerCase());
  const envFolder = on && !ON.has(raw.toLowerCase()) ? raw : null;
  if ((setting === undefined || setting === null) && !on) return null;
  if (setting === undefined || setting === null || setting === true) {
    const f = envFolder ?? defaultFolder();
    return f ? expand(f) : null;
  }
  return expand(setting);
}

const warned = new Set<string>();
export function warnOnce(key: string, message: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(`functai: ${message}`);
}

/** Who is calling: $FUNCTAI_CALLER, with the `caller` setting's keys over it. */
export function callerOf(settings: Settings): Rec {
  const raw = env()["FUNCTAI_CALLER"] ?? "";
  let base: Rec = {};
  if (raw) {
    try {
      const parsed = parseData(raw);
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) base = parsed as Rec;
      else throw new Error("not a JSON object");
    } catch (err) {
      warnOnce(`caller:${raw}`, `$FUNCTAI_CALLER is not a JSON object (${(err as Error).message}); ignored`);
    }
  }
  return lmcc.copyObject(base, settings.caller ?? {});             // members in order, as Python's {**env, **setting}
}

// ------------------------------------------------------------------ a call

export interface Program {
  name: string;
  kind: "ai" | "module" | "remote";
  /** A program served elsewhere: where (serving.md). */
  remote?: string;
  module: string;
  version: string;
  /** AI functions only: lmcc's signature fingerprint with every type name empty. */
  signature?: string;
  /** Every program: its interface's signature (programs.md). */
  interface: string;
  answer: string;
  saved?: string;
  file?: string;
  line?: number;
}

interface Exchange {
  model: string;
  provider: string | null;
  started: number;
  seconds: number;
  cached: boolean;
  request: Request;
  requestHash: string | null;
  response: Response | null;
  error: unknown;
  streamed: boolean;
  firstDelta: number | null;
}

/** One call being made: what its record and its events need. */
export class Call {
  readonly id: string = newId();
  readonly parent: string | null;
  readonly root: string;
  readonly started = now();
  readonly exchanges: Exchange[] = [];
  provider: string | null = null;
  /**
   * Its inputs as the record holds them, by its interface's names, each as
   * [JSON, size, whether it is a description]: written when the call starts
   * (left-out optional inputs of a module with no default are absent).
   */
  inputsJson: Record<string, readonly [unknown, number, boolean]> = {};
  /** Its outputs by name, once it has them. */
  outputs: Rec | null = null;
  confidence: number | null = null;
  /** What a required journal said of its end, when it did not confirm it. */
  journal: "refused" | "unknown" | null = null;
  /** Its requests so far (the next `request` event's number is one more). */
  requests = 0;
  /** Its place in its tree's log (set when it starts). */
  node!: Node;
  /**
   * Its place in its tree by names and order (contract/conversations.md,
   * `site`): the outermost call's name, then each child's, each with its
   * occurrence among its parent's children of that name (`support#1/answer#1`).
   */
  readonly path: string;
  /** How many children of each name it started (its children's sites). */
  readonly names = new Map<string, number>();
  /** Made while a tool ran: that tool call's number in the call that asked for it (tools.md). */
  invocation: number | null = null;
  /** The tool calls this call made so far (each one's invocation is the next). */
  invocations = 0;
  /** A conversation's turn: `{ id, turn, parent }` (its record's `conversation`). */
  conversation: Rec | null = null;
  /** A call continued by a later writer (a resumed turn): that writer's number. */
  writer: number | null = null;
  /** The conversation turn this call runs in (conversations.ts), or null. */
  turnRun: TurnHooks | null = null;
  /** What it was shown as context (its record's `saw`). */
  saw: Rec[] = [];
  /** What plugins changed (plugins.md, "Records"). */
  readonly changes: Rec[] = [];
  /** False when a plugin replaced a provider request. */
  replayable = true;
  /** Text its conversation's `context` hooks added to its instruction. */
  sections: string[] = [];
  /** An AI function's lmcc turn steps, when they are kept (calls.md, `steps`). */
  steps: unknown[] | null = null;
  /** A first model was unsure and another answered. */
  escalated = false;
  /** `{ output: { answer: probability } }`, when the model measured them. */
  probabilities: Record<string, Record<string, number>> | null = null;
  /** Tokens of its own exchanges, by kind. */
  get usage(): Record<string, number> {
    const usage: Record<string, number> = {};
    for (const e of this.exchanges) if (e.response) for (const [k, v] of Object.entries(usageOf(e.response))) usage[k] = (usage[k] ?? 0) + v;
    return usage;
  }
  /** The value its program gave (an AI function's prediction), once it has one: what a remembered helper's call keeps. */
  value: unknown = undefined;

  /** The caller's call in another process's log (a served call's `FunctAI-Parent`): its record's `parent`. */
  set remoteParent(id: string) {
    (this as { parent: string | null }).parent = id;
  }

  readonly program: () => Program;
  readonly folder: string | null;
  /** Its fields (interface inputs and outputs, and those FunctAI adds), for what the log keeps. */
  readonly fields: CallFields;
  /** Whether each field's value is written (calls.md, "Content"). */
  readonly keep: Record<string, boolean>;
  readonly caller: Rec;
  /** Cancels the call (its caller's signal, a stream closed, a parent cancelled). */
  readonly signal: AbortSignal | undefined;

  constructor(program: () => Program, parent: Call | undefined, folder: string | null, fields: CallFields,
    keep: Record<string, boolean>, caller: Rec, signal?: AbortSignal) {
    this.program = program;
    this.folder = folder;
    this.fields = fields;
    this.keep = keep;
    this.caller = caller;
    this.signal = signal;
    this.parent = parent?.id ?? null;
    this.root = parent?.root ?? this.id;
    const name = program().name;
    if (parent) {
      const n = (parent.names.get(name) ?? 0) + 1;
      parent.names.set(name, n);
      this.path = `${parent.path}/${name}#${n}`;
    } else this.path = `${name}#1`;
  }

  /** The call's id, minted by its turn when it is a conversation's turn (known before the call starts). */
  set mintedId(id: string) {
    (this as { id: string }).id = id;
    if (this.parent === null) (this as { root: string }).root = id;
  }

  /** Whether every value is written. */
  get content(): boolean {
    return Object.values(this.keep).every(Boolean);
  }

  exchange(model: string, request: Request, response: Response | null, started: number, seconds: number,
    opts: { cached?: boolean; error?: unknown; streamed?: boolean; firstDelta?: number | null; requestHash?: string | null } = {}): void {
    this.exchanges.push({
      model, provider: this.provider, started, seconds, cached: opts.cached ?? false, request, response,
      requestHash: opts.requestHash ?? null, error: opts.error ?? null, streamed: opts.streamed ?? false, firstDelta: opts.firstDelta ?? null,
    });
  }

  /** Make one of its events (streaming.md) when anything watches it: its streams, observers, the tree's journal. */
  event(kind: "request" | "text" | "thinking" | "tool_call" | "tool_result" | "retry" | "approval" | "approved", fields: Rec) {
    return this.node?.log.emit(this.node, kind, fields) ?? null;
  }

  /** Whether anything watches this call's events. */
  get watched(): boolean {
    return this.node ? this.node.log.watched(this.node) : false;
  }
}

export const current = new Context<Call>();
/** A call made while a tool runs: the number of that tool call in the call that asked (engine.ts sets it). */
export const INVOCATION = new Context<number | null>();
/** The call of another process a served call is made for (`FunctAI-Parent`): the outermost call's `parent` (serving.md). */
export const REMOTE_PARENT = new Context<{ id: string | null }>();

/**
 * What a call needs from the conversation turn it runs in (conversations.ts):
 * the replies, tool results and answers a resumed turn recorded, and the
 * records a turn writes as it goes (contract/tools.md, *Resuming*).
 */
export interface TurnHooks {
  /** The turn's id, which is its own call's. */
  readonly turn: string;
  /** Whether its records keep what resuming needs (a module's turn, or an AI function's with tools). */
  readonly durable: boolean;
  recordedReply(key: string): Response | null;
  noteReply(key: string, response: Response): void;
  /** A tool's result kept by an earlier attempt; throws `turn-unfinished` for one that may have run. */
  recordedTool(call: Call, approval: Approval): string | null;
  toolStarted(call: Call, approval: Approval): void;
  toolDone(call: Call, approval: Approval, output: string): void;
  /** `[allowed, reason, by, fresh]`: an answer recorded for this question (`fresh`: given while the turn waited). */
  recordedApproval(call: Call, approval: Approval): [boolean, string | null, string | null, boolean] | null;
  noteApproval(call: Call, approval: Approval, allowed: boolean, reason: string | null, by: string | null): void;
  /** Stop the turn to wait for a person (throws). */
  pause(call: Call, approval: Approval): never;
  /** A call of this turn starts: the turn's own (attach it), or one inside. */
  started(call: Call): void;
  /** A call of this turn ended. */
  ended(call: Call, failed: boolean): void;
}

/** A conversation's turn being run by this process, as a call sees it (conversations.ts makes them). */
export interface TurnRunLike extends TurnHooks {
  /** The program the turn is with: its call is the turn's own. */
  readonly self: object;
  /** The turn's own call has started. */
  rootTaken: boolean;
  /** Aborted when the turn is stopped (from any process) or taken over. */
  readonly signal: AbortSignal;
  /** For a resumed turn's own call: this process's claim on its log. */
  laterWriter(call: Call): { writer: number; after: { writer: number; seq: number } | null; at: string; requests: number } | null;
  /** An observer that keeps the turn's kept log in its store (another process watches it there), or null. */
  eventsSink(call: Call): Observer | null;
}

/** The turn being started in this context (conversations.ts sets it while the turn's call begins). */
export const ACTIVE_TURN = new Context<TurnRunLike | null>();

/** FunctAI's own refusal codes (contract/README.md), recorded with their errors. */
const OWN_CODES = /^(interface-|log-content-|journal-|saved-|event-)[a-z-]+$/;

/** An error as a record or an event holds it: its type, its message, and its code when it has one. */
export function errorJson(err: unknown, content = true): Rec {
  const e = err as { name?: string; code?: unknown; message?: string; constructor?: { name?: string } };
  const out: Rec = { type: e?.constructor?.name && e.constructor.name !== "Error" ? e.constructor.name : (e?.name ?? "Error") };
  if (typeof e?.code === "string" && (lmcc.isRefusal(err) || err instanceof FunctAIError || OWN_CODES.test(e.code))) out["code"] = e.code;
  if (content) out["message"] = e?.message ?? String(err);
  return out;
}

function usageOf(response: Response): Record<string, number> {
  const u = (Response.toJSON(response)["usage"] ?? {}) as Rec;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(u)) if (typeof v === "number" && Number.isInteger(v)) setOwn(out, k, v);
  return out;
}

const round6 = (x: number) => Math.round(x * 1e6) / 1e6;

function exchangeJson(ex: Exchange, content: boolean): Rec {
  const out: Rec = { model: ex.model, provider: ex.provider, started: iso(ex.started), seconds: round6(ex.seconds), cached: ex.cached };
  if (ex.streamed) {
    out["streamed"] = true;
    out["first_delta"] = ex.firstDelta === null ? null : round6(ex.firstDelta);
  }
  if (ex.response) {
    out["finish"] = ex.response.finishReason;
    out["usage"] = usageOf(ex.response);
  }
  if (ex.error !== null && ex.error !== undefined) out["error"] = errorJson(ex.error, content);
  if (content) {
    out["request"] = Request.toJSON(ex.request);
    if (ex.requestHash) out["request_hash"] = ex.requestHash;
    if (ex.response) out["response"] = Response.toJSON(ex.response);
  }
  return out;
}

let processInfo: Rec | null = null;
function processJson(): Rec {
  if (!processInfo) {
    const os = builtin("node:os");
    let user: string | null = null;
    try {
      user = os ? os.userInfo().username : null;
    } catch {
      user = null;
    }
    processInfo = { host: os ? os.hostname() : null, pid: pid(), user, language: "typescript", runtime: runtime(), functai: VERSION };
    // the libraries that build the request and send it (calls.md, process), when their versions can be read
    for (const [key, v] of [["lmcc", LMCC_VERSION], ["lm15", LM15_VERSION]] as const) if (typeof v === "string" && v) processInfo[key] = v;
  }
  return { ...processInfo };
}

/**
 * The call's record (contract/calls.md, "A call record"), format 2: the
 * whole record, then what `logContent` lets it keep.
 */
/** How a call ended, for its record: what it returned, or what it threw (anything, `undefined` included). */
export type Ending = { readonly failed: false; readonly returned: unknown } | { readonly failed: true; readonly error: unknown };

export function record(call: Call, ending: Ending): Rec {
  const program = call.program();
  const described: { inputs: string[]; outputs: string[] } = { inputs: [], outputs: [] };
  const written = (from: Record<string, readonly [unknown, number, boolean]>, which: "inputs" | "outputs"): [Rec, Record<string, number>] => {
    const data: Rec = {};
    const sizes: Record<string, number> = {};
    for (const [k, [json, n, isDescription]] of entriesOf(from)) {
      setOwn(data, k, json);
      setOwn(sizes, k, n);
      if (isDescription) described[which].push(k);
    }
    return [data, sizes];
  };
  const values = (from: Rec, which: "inputs" | "outputs") =>
    written(recordOf(entriesOf(from).map(([k, v]) => [k, toJson(v)] as const)), which);
  const [inputs, inSizes] = written(call.inputsJson, "inputs");
  const failed = ending.failed;
  const [outputs, outSizes] = call.outputs && !failed ? values(call.outputs, "outputs") : [null, {}];
  const answered = call.exchanges.filter((e) => e.response !== null);
  const usage: Record<string, number> = {};
  for (const e of answered) for (const [k, v] of Object.entries(usageOf(e.response!))) setOwn(usage, k, (getOwn(usage, k) ?? 0) + v);
  const rec: Rec = {
    functai_call: FORMAT, id: call.id, parent: call.parent, root: call.root, program,
    started: iso(call.started), seconds: round6((now() - call.started) / 1000), content: true,
    inputs, outputs,
  };
  if (program.kind === "ai" && !ending.failed) {
    const [shown] = toJson(ending.returned);
    if (outputs === null || !Object.hasOwn(outputs, program.answer) || !lmcc.jsonEqual(getOwn(outputs as Rec, program.answer) as lmcc.Json, shown)) rec["returned"] = shown;
  }
  rec["sizes"] = { inputs: inSizes, outputs: outSizes };
  if (described.inputs.length || described.outputs.length) rec["described"] = described;
  rec["error"] = ending.failed ? errorJson(ending.error, true) : null;
  rec["model"] = answered.length ? answered[answered.length - 1]!.model : null;
  rec["usage"] = usage;
  rec["confidence"] = call.confidence;
  if (call.probabilities && Object.keys(call.probabilities).length) rec["probabilities"] = copyData(call.probabilities);
  if (call.escalated) rec["escalated"] = true;
  rec["exchanges"] = call.exchanges.map((e) => exchangeJson(e, true));
  if (call.steps !== null) rec["steps"] = copyData(call.steps);
  rec["saw"] = copyData(call.saw);
  if (call.invocation !== null) rec["invocation"] = call.invocation;
  if (call.conversation !== null) rec["conversation"] = copyData(call.conversation);
  if (call.writer !== null && call.writer > 1) rec["writer"] = call.writer;
  if (call.sections.length) rec["sections"] = [...call.sections];
  if (call.changes.length) rec["changes"] = copyData(call.changes);
  if (!call.replayable) rec["replayable"] = false;
  if (call.journal) rec["journal"] = call.journal;
  rec["caller"] = lmcc.copyObject(call.caller);
  rec["process"] = processJson();
  return writtenRecord(rec, call.fields, call.keep);
}

function line(rec: Rec): string {
  // numbers as they came (a temperature of 0.0 stays 0.0) and every value's members in its order, as Python writes them
  const dump = (r: Rec) => writeData(r) + "\n";
  let text = dump(rec);
  if (new TextEncoder().encode(text).length <= MAX_LINE) return text;
  rec = { ...rec, truncated: true, exchanges: (rec["exchanges"] as Rec[]).map(({ request: _q, response: _r, ...rest }) => rest) };
  text = dump(rec);
  if (new TextEncoder().encode(text).length <= MAX_LINE) return text;
  for (const k of ["inputs", "outputs", "returned", "probabilities"]) delete rec[k];
  return dump(rec);
}

const fileName = (() => {
  let name: string | null = null;
  return () => {
    if (!name) {
      const os = builtin("node:os");
      const rand = Array.from(globalThis.crypto.getRandomValues(new Uint8Array(3)), (b) => b.toString(16).padStart(2, "0")).join("");
      name = `${os ? os.hostname() : "host"}-${pid() ?? 0}-${rand}.jsonl`;
    }
    return name;
  };
})();

/** Append one line to this process's file in `folder` (one per UTC day). */
export function append(folder: string, rec: Rec): void {
  const fs = builtin("node:fs");
  const path = builtin("node:path");
  if (!fs || !path) {
    warnOnce("no-fs", "this runtime has no file system: calls are not logged");
    return;
  }
  const day = path.join(folder, new Date().toISOString().slice(0, 10));
  fs.mkdirSync(day, { recursive: true, mode: 0o700 });
  fs.appendFileSync(path.join(day, fileName()), line(rec), { mode: 0o600 });
}

/** End a call: write its line when logging is on. Never throws into the call. */
export function write(call: Call, ending: Ending): void {
  if (!call.folder) return;
  try {
    append(call.folder, record(call, ending));
  } catch (err) {
    warnOnce(`${call.folder}:${(err as Error).name}`,
      `could not log a call of ${call.program().name} to ${call.folder} (${(err as Error).message}); calls go on, unlogged`);
  }
}

// ------------------------------------------------------------------ reading

/** The `.jsonl` files a reader reads: the top level (kept files, calls.md "The folder"), then each day's from `since`'s day. */
function logFiles(root: string, since: string): string[] {
  const fs = builtin("node:fs")!;
  const path = builtin("node:path")!;
  if (!fs.existsSync(root)) return [];
  const names = fs.readdirSync(root).sort();
  const out = names.filter((f) => f.endsWith(".jsonl")).map((f) => path.join(root, f)).filter((f) => fs.statSync(f).isFile());
  for (const day of names) {
    const dir = path.join(root, day);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || !fs.statSync(dir).isDirectory()) continue;
    if (since && day < since.slice(0, 10)) continue;
    for (const file of fs.readdirSync(dir).sort()) if (file.endsWith(".jsonl")) out.push(path.join(dir, file));
  }
  return out;
}

function linesOf(file: string): Rec[] {
  const fs = builtin("node:fs")!;
  let text: string;
  try {
    text = fs.readFileSync(file, "utf8");
  } catch {
    return [];
  }
  const out: Rec[] = [];
  for (const raw of text.split("\n")) {
    if (!raw.trim()) continue;
    let rec: unknown;
    try {
      rec = parseData(raw);
    } catch {
      continue;
    }
    if (typeof rec === "object" && rec !== null && !Array.isArray(rec)) out.push(rec as Rec);
  }
  return out;
}

/** A time to read from: a `Date`, or text like `"7d"`, `"12h"`, `"2w"` or a date (`"2026-09-20"`). */
export function sinceOf(since: Date | string | null | undefined): Date | null {
  if (since === null || since === undefined) return null;
  if (since instanceof Date) return since;
  const m = /^\s*(\d+)\s*([hdw])\s*$/.exec(since);
  if (m) return new Date(Date.now() - Number(m[1]) * { h: 3600e3, d: 86400e3, w: 7 * 86400e3 }[m[2] as "h" | "d" | "w"]);
  const d = new Date(since);
  if (Number.isNaN(d.getTime())) throw new TypeError(`since is a Date, or text like "2026-09-20", "7d", "12h", "2w"; not ${JSON.stringify(since)}`);
  return d;
}

/**
 * (calls, ratings) logged in a folder, as the objects of their lines: the
 * folder's top-level files (what `pruneCalls` kept) and each day's. A call
 * continued by a later writer (a resumed turn) is the record of its highest
 * `writer` (calls.md).
 */
export function read(folder?: string | null, opts: { since?: Date | string | null } = {}): [Rec[], Rec[]] {
  const fs = builtin("node:fs");
  const path = builtin("node:path");
  const root = folder ? expand(folder) : folderOf(true);
  if (!fs || !path || !root) throw new Error("reading a log folder needs a file system; pass the records instead");
  const start = sinceOf(opts.since);
  const cutoff = start ? iso(start.getTime()) : "";
  const calls: Rec[] = [];
  const ratings: Rec[] = [];
  for (const file of logFiles(root, cutoff)) {
    for (const r of linesOf(file)) {
      if (CALL_FORMATS.has(r["functai_call"] as number) && String(r["started"] ?? "") >= cutoff) calls.push(r);
      else if (r["functai_rating"] === RATING_FORMAT && String(r["at"] ?? "") >= cutoff) ratings.push(r);
    }
  }
  return [latestWriters(calls), ratings];
}

/** One record per call: a call continued by a later writer is the record of its highest `writer`. */
export function latestWriters(calls: readonly Rec[]): Rec[] {
  const best = new Map<unknown, Rec>();
  for (const c of calls) {
    const had = best.get(c["id"]);
    if (!had || Number(c["writer"] ?? 1) >= Number(had["writer"] ?? 1)) best.set(c["id"], c);
  }
  const seen = new Set<unknown>();
  const out: Rec[] = [];
  for (const c of calls) {
    if (seen.has(c["id"])) continue;
    seen.add(c["id"]);
    out.push(best.get(c["id"])!);
  }
  return out;
}

/**
 * Delete the call log's day folders older than a time, keeping what ratings
 * need (calls.md, "The folder"). Before a day goes, every rated call in it,
 * every call of its tree, every call its `saw` names and their ratings are
 * copied into one file at the folder's top level
 * (`kept-<host>-<pid>-<hex>.jsonl`), which every reader reads: a row of
 * `rated` made before pruning is made the same after. Returns how many day
 * folders and calls went, and how many calls were kept.
 */
export function pruneCalls(opts: { olderThan?: Date | string; folder?: string; keepRated?: boolean } = {}): { days: number; calls: number; kept: number } {
  const fs = builtin("node:fs");
  const path = builtin("node:path");
  const os = builtin("node:os");
  if (!fs || !path) throw new Error("pruning a log folder needs a file system");
  const root = opts.folder ? expand(opts.folder) : folderOf(true);
  const start = sinceOf(opts.olderThan ?? "90d");
  if (!root || !start) throw new TypeError("olderThan is a time: \"90d\", a date");
  const first = iso(start.getTime()).slice(0, 10);
  if (!fs.existsSync(root)) return { days: 0, calls: 0, kept: 0 };
  const oldDays = fs.readdirSync(root).sort().filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d) && d < first && fs.statSync(path.join(root, d)).isDirectory());
  if (!oldDays.length) return { days: 0, calls: 0, kept: 0 };
  const [everything, ratings] = read(root);
  const byId = new Map(everything.map((c) => [c["id"] as string, c]));
  const inOld = new Set<string>();
  const oldLines: Rec[] = [];
  for (const day of oldDays) {
    for (const file of fs.readdirSync(path.join(root, day)).sort()) {
      if (!file.endsWith(".jsonl")) continue;
      for (const r of linesOf(path.join(root, day, file))) {
        oldLines.push(r);
        if ("functai_call" in r) inOld.add(r["id"] as string);
      }
    }
  }
  const keep = new Set<string>();
  if (opts.keepRated !== false) {
    const rated = new Set(ratings.map((r) => r["call"] as string));
    const trees = new Set([...rated].filter((c) => byId.has(c)).map((c) => byId.get(c)!["root"]));
    for (const [id, c] of byId) if (rated.has(id) || trees.has(c["root"])) keep.add(id);
    const todo = [...keep];
    while (todo.length) {                          // every call a kept call's saw names, and theirs
      const c = byId.get(todo.pop()!);
      for (const entry of ((c?.["saw"] ?? []) as Rec[])) {
        if (typeof entry !== "object" || entry === null) continue;
        for (const key of ["call", "saw_of"]) {
          const id = entry[key] as string | undefined;
          if (id && byId.has(id) && !keep.has(id)) {
            keep.add(id);
            todo.push(id);
          }
        }
      }
    }
  }
  const keptLines = oldLines.filter((r) => ("functai_call" in r && keep.has(r["id"] as string)) || ("functai_rating" in r && keep.has(r["call"] as string)));
  if (keptLines.length) {
    const host = (os ? os.hostname() : "host").replace(/[^A-Za-z0-9_.-]/g, "_") || "host";
    const rand = Array.from(globalThis.crypto.getRandomValues(new Uint8Array(3)), (b) => b.toString(16).padStart(2, "0")).join("");
    const file = path.join(root, `kept-${host}-${pid() ?? 0}-${rand}.jsonl`);
    const fd = fs.openSync(file, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL, 0o600);
    try {
      fs.writeSync(fd, keptLines.map((r) => writeData(r) + "\n").join(""));
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
  }
  for (const day of oldDays) fs.rmSync(path.join(root, day), { recursive: true, force: true });
  const keptCalls = keptLines.filter((r) => "functai_call" in r).length;
  return { days: oldDays.length, calls: [...inOld].filter((id) => !keep.has(id)).length, kept: keptCalls };
}

const order = (r: Rec, key: string): string => `${r[key] ?? ""}\u0000${r["id"] ?? ""}`;
const later = (a: Rec, b: Rec, key: string) => {
  const ta = String(a[key] ?? ""), tb = String(b[key] ?? "");
  if (ta !== tb) return ta > tb;
  return String(a["id"] ?? "") > String(b["id"] ?? "");
};

/**
 * For each call, the ratings that count: each person's latest, none for a
 * withdrawn one. A rating with no `by` (made under an account, which may be
 * shared) counts on its own: it replaces none, none replaces it, and its
 * null verdict withdraws nothing (calls.md, rule 2).
 */
export function currentRatings(ratings: Iterable<Rec>, by?: string | null): Map<string, Rec[]> {
  const latest = new Map<string, Rec>();
  for (const r of ratings) {
    if (by !== undefined && by !== null && r["by"] !== by) continue;
    const person = typeof r["by"] === "string" && r["by"] ? r["by"] : null;
    const key = person !== null ? JSON.stringify([r["call"], "by", person]) : JSON.stringify([r["call"], "rating", r["id"]]);
    const had = latest.get(key);
    if (!had || later(r, had, "at")) latest.set(key, r);
  }
  const out = new Map<string, Rec[]>();
  for (const r of latest.values()) {
    if (r["verdict"] !== "right" && r["verdict"] !== "wrong") continue;
    const list = out.get(r["call"] as string) ?? [];
    list.push(r);
    out.set(r["call"] as string, list);
  }
  for (const list of out.values()) list.sort((a, b) => (later(a, b, "at") ? 1 : later(b, a, "at") ? -1 : 0));
  return out;
}

/** Whether a call's record kept every input as data (rule 3's `no_content`). */
function inputsKept(call: Rec): boolean {
  const described = ((call["described"] ?? {}) as Rec)["inputs"] as unknown[] | undefined;
  if (described?.length) return false;
  if (call["content"] === true) return "inputs" in call;
  const omitted = call["omitted"] as { inputs?: unknown[] } | undefined;
  return call["functai_call"] === 2 && omitted !== undefined && Array.isArray(omitted.inputs) && omitted.inputs.length === 0;
}

/** The values a counting rating gives (rule 4), or null. */
function says(rating: Rec, call: Rec): Rec | null {
  const answer = ((call["program"] ?? {}) as Rec)["answer"] as string || "result";
  if (rating["verdict"] === "right") {
    const outputs = (call["outputs"] ?? null) as Rec | null;
    const described = (((call["described"] ?? {}) as Rec)["outputs"] ?? []) as string[];
    return outputs && Object.hasOwn(outputs, answer) && !described.includes(answer) ? recordOf([[answer, outputs[answer]]]) : null;
  }
  const values: Rec = {};
  if (Object.hasOwn(rating, "answer")) setOwn(values, answer, rating["answer"]);
  for (const [k, v] of entriesOf((rating["outputs"] ?? {}) as Rec)) if (!Object.hasOwn(values, k)) setOwn(values, k, v);
  return Object.keys(values).length ? values : null;
}

function addMeta(row: Rec, meta: Rec): void {
  for (let [key, value] of Object.entries(meta)) {
    while (Object.hasOwn(row, key)) key = "_" + key;
    setOwn(row, key, value);
  }
}

export interface LeftOut { other_signature: number; no_content: number; no_answer: number; no_context?: number }

/** Rows with known answers from rated calls (contract/calls.md, "Rows with known answers"). */
export function ratedRows(calls: Iterable<Rec>, ratings: Iterable<Rec>,
  opts: { name: string; module?: string | null; file?: string | null; signature?: string | null; interface?: string | null; by?: string | null }): [Rec[], LeftOut] {
  const counting = currentRatings([...ratings].filter((r) => r["functai_rating"] === RATING_FORMAT), opts.by);
  const left: LeftOut = { other_signature: 0, no_content: 0, no_answer: 0 };
  const rows: Rec[] = [];
  const mine = [...calls].filter((c) => CALL_FORMATS.has(c["functai_call"] as number)).filter((c) => {
    const p = (c["program"] ?? {}) as Rec;
    // a program defined at the top level (a notebook, a script) is known by its file too: with a file, a call with none does not match
    return p["name"] === opts.name && (opts.module === undefined || opts.module === null || p["module"] === opts.module)
      && (opts.file === undefined || opts.file === null || p["file"] === opts.file);
  });
  mine.sort((a, b) => (order(a, "started") < order(b, "started") ? -1 : order(a, "started") > order(b, "started") ? 1 : 0));
  for (const call of mine) {
    const rs = counting.get(call["id"] as string);
    if (!rs || !rs.length) continue;
    const program = (call["program"] ?? {}) as Rec;
    const bySignature = opts.signature !== undefined && opts.signature !== null;
    const byInterface = opts.interface !== undefined && opts.interface !== null;
    if (bySignature || byInterface) {
      const own = (program["interface"] ?? program["signature"]) as string | undefined;
      const matches = (byInterface && own === opts.interface) || (bySignature && program["signature"] === opts.signature);
      if (!matches) {
        left.other_signature++;
        continue;
      }
    }
    if (!inputsKept(call)) {
      left.no_content++;
      continue;
    }
    const usable = rs.map((r) => [r, says(r, call)] as const).filter(([, v]) => v !== null) as [Rec, Rec][];
    if (!usable.length) {
      left.no_answer++;
      continue;
    }
    const [rating, values] = usable[usable.length - 1]!;
    const verdicts = new Set(rs.map((r) => r["verdict"]));
    const spelled = new Set(usable.map(([, v]) => lmcc.canonicalJson(v as lmcc.Json)));
    const disputed = verdicts.size > 1 || spelled.size > 1;
    const answer = (program["answer"] as string) || "result";
    const row: Rec = lmcc.copyObject((call["inputs"] ?? {}) as Rec);
    if (Object.hasOwn(values, answer)) setOwn(row, answer, values[answer]);
    for (const [k, v] of entriesOf(values)) if (k !== answer) setOwn(row, k, v);
    addMeta(row, {
      call: call["id"], version: program["version"], rating: rating["verdict"], rated_by: rating["by"] ?? null,
      origin: rating["origin"] ?? "review", sample: rating["sample"] ?? null, disputed,
    });
    rows.push(row);
  }
  return [rows, left];
}

/** A column `ratedRows` added, under its name or with underscores in front. */
export function metaOf(row: Rec, key: string): unknown {
  const keys = Object.keys(row).reverse();
  const k = keys.find((x) => x.replace(/^_+/, "") === key);
  return k === undefined ? undefined : row[k];
}

const META = 7;

/**
 * Rows of calls that were shown earlier turns (a conversation), each with
 * `earlier`, `conversation` (and `sections`, `helpers`) after the data and
 * before the rating's columns, and how many rows were left out because the
 * log cannot show those turns again (`no_context`). When no row was shown
 * any, the rows are as they were (calls.md, "Rows that keep their context").
 */
export function withContext(rows: readonly Rec[], calls: Iterable<Rec>): [Rec[], number] {
  const by = recordsById([...calls]);
  if (!rows.some((r) => needsContext(by.get(metaOf(r, "call") as string), by))) return [[...rows], 0];
  const out: Rec[] = [];
  let dropped = 0;
  for (const r of rows) {
    let ctx: Rec;
    try {
      ctx = earlierOf(by, metaOf(r, "call") as string);
    } catch (err) {
      if (err instanceof SawUnknown) {
        dropped++;
        continue;
      }
      throw err;
    }
    const keys = Object.keys(r);
    const rebuilt = recordOf(entriesOf(r).slice(0, keys.length - META));
    addMeta(rebuilt, ctx);
    for (const [k, v] of entriesOf(r).slice(keys.length - META)) setOwn(rebuilt, k, v);
    out.push(rebuilt);
  }
  return [out, dropped];
}

/**
 * Two lists, with every group of rows on one side: `const [train, test] =
 * split(rated(tutor).rows)`. Turns of one conversation depend on each other:
 * a test row whose conversation is also in the training rows measures
 * memory, not the program. A row with no group (`conversation` null) is a
 * group of its own. `test` is the share of groups held out (at least one
 * group each side when there are two or more); `seed` makes it repeatable
 * here (not Python's draw for the same seed).
 */
export function split<R extends Rec>(rows: readonly R[], opts: { by?: string; test?: number; seed?: number } = {}): [R[], R[]] {
  const by = opts.by ?? "conversation";
  const test = opts.test ?? 0.2;
  if (!(test > 0 && test < 1)) throw new RangeError(`test is a share between 0 and 1, not ${test}`);
  const groups: string[] = [];
  const keyOf: string[] = [];
  rows.forEach((r, i) => {
    if (!Object.hasOwn(r, by)) throw new TypeError(`the rows have no column ${JSON.stringify(by)}`);
    const v = r[by];
    const g = v === null || v === undefined ? `\u0000row:${i}` : lmcc.canonicalJson(v as lmcc.Json);
    keyOf.push(g);
    if (!groups.includes(g)) groups.push(g);
  });
  let state = (opts.seed ?? 0) >>> 0;
  const rng = () => {                                          // mulberry32
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = groups.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [groups[i], groups[j]] = [groups[j]!, groups[i]!];
  }
  const nTest = groups.length > 1 ? Math.min(Math.max(1, Math.round(groups.length * test)), groups.length - 1) : 0;
  const held = new Set(groups.slice(0, nTest));
  return [rows.filter((_, i) => !held.has(keyOf[i]!)), rows.filter((_, i) => held.has(keyOf[i]!))];
}

// ------------------------------------------------------------------ rating

export type Verdict = "right" | "wrong" | true | false | null;

export interface RateOptions {
  answer?: unknown;
  outputs?: Rec;
  note?: string;
  reasons?: string[];
  by?: string;
  origin?: "review" | "edit";
  sample?: string;
  folder?: string;
}

/** A rating record (contract/calls.md, "A rating record"), written to the log when there is a folder. */
export function rating(callId: string, verdict: Verdict | undefined, opts: RateOptions, settings: Settings): Rec {
  const corrected = "answer" in opts || (opts.outputs !== undefined && Object.keys(opts.outputs).length > 0);
  let v: "right" | "wrong" | null;
  if (verdict === undefined) {
    if (!corrected) throw new TypeError("say whether the answer is right: rate(p, \"right\"), rate(p, \"wrong\"), or rate(p, undefined, { answer })");
    v = "wrong";
  } else if (verdict === null) v = null;
  else if (verdict === true || verdict === "right") v = "right";
  else if (verdict === false || verdict === "wrong") v = "wrong";
  else throw new TypeError(`verdict is "right", "wrong", true, false, or null (withdraw); not ${JSON.stringify(verdict)}`);
  if (corrected && v !== "wrong") throw new TypeError("a right answer needs no correction: give answer or outputs only with \"wrong\"");
  // a person when one was named; else the account, which names no one (it may be shared: calls.md, "A rating record")
  const person = opts.by ?? (callerOf(settings)["user"] as string | undefined);
  const who: Rec = person ? { by: person } : { account: (processJson()["user"] as string | null) ?? "unknown" };
  const rec: Rec = { functai_rating: RATING_FORMAT, id: newId(), call: callId, at: iso(Date.now()), ...who, verdict: v };
  if ("answer" in opts) rec["answer"] = toJson(opts.answer)[0];
  if (opts.outputs) rec["outputs"] = recordOf(entriesOf(opts.outputs).map(([k, x]) => [k, toJson(x)[0]] as const));
  if (opts.reasons?.length) rec["reasons"] = [...opts.reasons];
  if (opts.note) rec["note"] = opts.note;
  rec["origin"] = opts.origin ?? "review";
  if (opts.sample) rec["sample"] = opts.sample;
  const folder = opts.folder ? expand(opts.folder) : folderOf(settings.logCalls ?? true);
  if (!folder) throw new Error("no log folder: pass folder, or turn the call log on (logCalls, or FUNCTAI_LOG_CALLS)");
  append(folder, rec);
  return rec;
}
