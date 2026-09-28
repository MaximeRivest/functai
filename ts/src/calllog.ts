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
import { Request, Response, stringifyJson } from "@lm15/lm15";
import { writtenRecord, type CallFields } from "./content.ts";
import type { Node } from "./log.ts";
import { builtin, Context, env, pid, runtime } from "./host.ts";
import type { Settings } from "./settings.ts";
import { setOwn, toJson } from "./values.ts";

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
      const parsed = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) base = parsed;
      else throw new Error("not a JSON object");
    } catch (err) {
      warnOnce(`caller:${raw}`, `$FUNCTAI_CALLER is not a JSON object (${(err as Error).message}); ignored`);
    }
  }
  return { ...base, ...(settings.caller ?? {}) };
}

// ------------------------------------------------------------------ a call

export interface Program {
  name: string;
  kind: "ai" | "module";
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
  readonly id = newId();
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
  event(kind: "request" | "text" | "thinking" | "tool_call" | "tool_result" | "retry", fields: Rec) {
    return this.node?.log.emit(this.node, kind, fields) ?? null;
  }

  /** Whether anything watches this call's events. */
  get watched(): boolean {
    return this.node ? this.node.log.watched(this.node) : false;
  }
}

export const current = new Context<Call>();

/** FunctAI's own refusal codes (contract/README.md), recorded with their errors. */
const OWN_CODES = /^(interface-|log-content-|journal-|saved-|event-)[a-z-]+$/;

/** An error as a record or an event holds it: its type, its message, and its code when it has one. */
export function errorJson(err: unknown, content = true): Rec {
  const e = err as { name?: string; code?: unknown; message?: string; constructor?: { name?: string } };
  const out: Rec = { type: e?.constructor?.name && e.constructor.name !== "Error" ? e.constructor.name : (e?.name ?? "Error") };
  if (typeof e?.code === "string" && (lmcc.isRefusal(err) || OWN_CODES.test(e.code))) out["code"] = e.code;
  if (content) out["message"] = e?.message ?? String(err);
  return out;
}

function usageOf(response: Response): Record<string, number> {
  const u = (Response.toJSON(response)["usage"] ?? {}) as Rec;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(u)) if (typeof v === "number" && Number.isInteger(v)) out[k] = v;
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
  }
  return { ...processInfo };
}

/**
 * The call's record (contract/calls.md, "A call record"), format 2: the
 * whole record, then what `logContent` lets it keep.
 */
export function record(call: Call, opts: { returned?: unknown; error?: unknown; hasReturned?: boolean } = {}): Rec {
  const program = call.program();
  const described: { inputs: string[]; outputs: string[] } = { inputs: [], outputs: [] };
  const written = (from: Record<string, readonly [unknown, number, boolean]>, which: "inputs" | "outputs"): [Rec, Record<string, number>] => {
    const data: Rec = {};
    const sizes: Record<string, number> = {};
    for (const [k, [json, n, isDescription]] of Object.entries(from)) {
      setOwn(data, k, json);
      setOwn(sizes, k, n);
      if (isDescription) described[which].push(k);
    }
    return [data, sizes];
  };
  const values = (from: Rec, which: "inputs" | "outputs") =>
    written(Object.fromEntries(Object.entries(from).map(([k, v]) => [k, toJson(v)])), which);
  const [inputs, inSizes] = written(call.inputsJson, "inputs");
  const failed = opts.error !== undefined;
  const [outputs, outSizes] = call.outputs && !failed ? values(call.outputs, "outputs") : [null, {}];
  const answered = call.exchanges.filter((e) => e.response !== null);
  const usage: Record<string, number> = {};
  for (const e of answered) for (const [k, v] of Object.entries(usageOf(e.response!))) usage[k] = (usage[k] ?? 0) + v;
  const rec: Rec = {
    functai_call: FORMAT, id: call.id, parent: call.parent, root: call.root, program,
    started: iso(call.started), seconds: round6((now() - call.started) / 1000), content: true,
    inputs, outputs,
  };
  if (program.kind === "ai" && opts.hasReturned && !failed) {
    const [shown] = toJson(opts.returned);
    if (outputs === null || !lmcc.jsonEqual((outputs as Rec)[program.answer] as lmcc.Json, shown)) rec["returned"] = shown;
  }
  rec["sizes"] = { inputs: inSizes, outputs: outSizes };
  if (described.inputs.length || described.outputs.length) rec["described"] = described;
  rec["error"] = failed ? errorJson(opts.error, true) : null;
  rec["model"] = answered.length ? answered[answered.length - 1]!.model : null;
  rec["usage"] = usage;
  rec["confidence"] = call.confidence;
  rec["exchanges"] = call.exchanges.map((e) => exchangeJson(e, true));
  rec["saw"] = [];
  if (call.journal) rec["journal"] = call.journal;
  rec["caller"] = { ...call.caller };
  rec["process"] = processJson();
  return writtenRecord(rec, call.fields, call.keep);
}

function line(rec: Rec): string {
  // lm15 writes its numbers as they came (a temperature of 0.0 stays 0.0), as Python does
  const dump = (r: Rec) => stringifyJson(r) + "\n";
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
export function write(call: Call, opts: { returned?: unknown; error?: unknown; hasReturned?: boolean }): void {
  if (!call.folder) return;
  try {
    append(call.folder, record(call, opts));
  } catch (err) {
    warnOnce(`${call.folder}:${(err as Error).name}`,
      `could not log a call of ${call.program().name} to ${call.folder} (${(err as Error).message}); calls go on, unlogged`);
  }
}

// ------------------------------------------------------------------ reading

/** (calls, ratings) logged in a folder, as the objects of their lines. */
export function read(folder?: string | null, opts: { since?: Date | null } = {}): [Rec[], Rec[]] {
  const fs = builtin("node:fs");
  const path = builtin("node:path");
  const root = folder ? expand(folder) : folderOf(true);
  if (!fs || !path || !root) throw new Error("reading a log folder needs a file system; pass the records instead");
  const cutoff = opts.since ? iso(opts.since.getTime()) : "";
  const calls: Rec[] = [];
  const ratings: Rec[] = [];
  if (!fs.existsSync(root)) return [calls, ratings];
  for (const day of fs.readdirSync(root).sort()) {
    const dir = path.join(root, day);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || !fs.statSync(dir).isDirectory()) continue;
    if (cutoff && day < cutoff.slice(0, 10)) continue;
    for (const file of fs.readdirSync(dir).sort()) {
      if (!file.endsWith(".jsonl")) continue;
      let text: string;
      try {
        text = fs.readFileSync(path.join(dir, file), "utf8");
      } catch {
        continue;
      }
      for (const raw of text.split("\n")) {
        if (!raw.trim()) continue;
        let rec: unknown;
        try {
          rec = JSON.parse(raw);
        } catch {
          continue;
        }
        if (typeof rec !== "object" || rec === null || Array.isArray(rec)) continue;
        const r = rec as Rec;
        if (CALL_FORMATS.has(r["functai_call"] as number) && String(r["started"] ?? "") >= cutoff) calls.push(r);
        else if (r["functai_rating"] === RATING_FORMAT && String(r["at"] ?? "") >= cutoff) ratings.push(r);
      }
    }
  }
  return [calls, ratings];
}

const order = (r: Rec, key: string): string => `${r[key] ?? ""}\u0000${r["id"] ?? ""}`;
const later = (a: Rec, b: Rec, key: string) => {
  const ta = String(a[key] ?? ""), tb = String(b[key] ?? "");
  if (ta !== tb) return ta > tb;
  return String(a["id"] ?? "") > String(b["id"] ?? "");
};

/** For each call, the ratings that count: each person's latest, none for a withdrawn one. */
export function currentRatings(ratings: Iterable<Rec>, by?: string | null): Map<string, Rec[]> {
  const latest = new Map<string, Rec>();
  for (const r of ratings) {
    if (by !== undefined && by !== null && r["by"] !== by) continue;
    const key = JSON.stringify([r["call"], r["by"]]);
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
    return outputs && answer in outputs && !described.includes(answer) ? { [answer]: outputs[answer] } : null;
  }
  const values: Rec = {};
  if ("answer" in rating) values[answer] = rating["answer"];
  for (const [k, v] of Object.entries((rating["outputs"] ?? {}) as Rec)) if (!(k in values)) values[k] = v;
  return Object.keys(values).length ? values : null;
}

function addMeta(row: Rec, meta: Rec): void {
  for (let [key, value] of Object.entries(meta)) {
    while (key in row) key = "_" + key;
    row[key] = value;
  }
}

export interface LeftOut { other_signature: number; no_content: number; no_answer: number }

/** Rows with known answers from rated calls (contract/calls.md, "Rows with known answers"). */
export function ratedRows(calls: Iterable<Rec>, ratings: Iterable<Rec>,
  opts: { name: string; module?: string | null; signature?: string | null; interface?: string | null; by?: string | null }): [Rec[], LeftOut] {
  const counting = currentRatings([...ratings].filter((r) => r["functai_rating"] === RATING_FORMAT), opts.by);
  const left: LeftOut = { other_signature: 0, no_content: 0, no_answer: 0 };
  const rows: Rec[] = [];
  const mine = [...calls].filter((c) => CALL_FORMATS.has(c["functai_call"] as number)).filter((c) => {
    const p = (c["program"] ?? {}) as Rec;
    return p["name"] === opts.name && (opts.module === undefined || opts.module === null || p["module"] === opts.module);
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
    const row: Rec = { ...((call["inputs"] ?? {}) as Rec) };
    if (answer in values) row[answer] = values[answer];
    for (const [k, v] of Object.entries(values)) if (k !== answer) row[k] = v;
    addMeta(row, {
      call: call["id"], version: program["version"], rating: rating["verdict"], rated_by: rating["by"],
      origin: rating["origin"] ?? "review", sample: rating["sample"] ?? null, disputed,
    });
    rows.push(row);
  }
  return [rows, left];
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
  const who = opts.by ?? (callerOf(settings)["user"] as string | undefined) ?? (processJson()["user"] as string | null) ?? "someone";
  const rec: Rec = { functai_rating: RATING_FORMAT, id: newId(), call: callId, at: iso(Date.now()), by: who, verdict: v };
  if ("answer" in opts) rec["answer"] = toJson(opts.answer)[0];
  if (opts.outputs) rec["outputs"] = Object.fromEntries(Object.entries(opts.outputs).map(([k, x]) => [k, toJson(x)[0]]));
  if (opts.reasons?.length) rec["reasons"] = [...opts.reasons];
  if (opts.note) rec["note"] = opts.note;
  rec["origin"] = opts.origin ?? "review";
  if (opts.sample) rec["sample"] = opts.sample;
  const folder = opts.folder ? expand(opts.folder) : folderOf(settings.logCalls ?? true);
  if (!folder) throw new Error("no log folder: pass folder, or turn the call log on (logCalls, or FUNCTAI_LOG_CALLS)");
  append(folder, rec);
  return rec;
}
