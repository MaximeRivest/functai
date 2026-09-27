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
import { builtin, Context, env, pid, runtime } from "./host.ts";
import type { Settings } from "./settings.ts";

export const FORMAT = 1;
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

/** A value as the JSON the log holds, and its size (code points of its canonical JSON). */
export function toJson(value: unknown): [Json, number] {
  let data: Json;
  try {
    data = plain(value);
    return [data, [...lmcc.canonicalJson(data as lmcc.Json)].length];
  } catch {
    const text = String(value).slice(0, 2000);
    data = { $type: typeName(value), $repr: text };
    return [data, [...lmcc.canonicalJson(data as lmcc.Json)].length];
  }
}

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (typeof v === "object") return (v as object).constructor?.name ?? "Object";
  return typeof v;
}

function plain(v: unknown): Json {
  if (v === null || typeof v === "string" || typeof v === "boolean") return v;
  if (typeof v === "number") {
    if (!Number.isFinite(v)) throw new TypeError("not finite");
    return v;
  }
  if (typeof v === "bigint") {
    if (v <= BigInt(Number.MAX_SAFE_INTEGER) && v >= BigInt(-Number.MAX_SAFE_INTEGER)) return Number(v);
    throw new TypeError("bigint");
  }
  if (v instanceof Date) return v.toISOString();
  if (Array.isArray(v)) return v.map(plain);
  if (typeof v === "object") {
    const proto = Object.getPrototypeOf(v);
    if (proto !== Object.prototype && proto !== null) throw new TypeError("not plain data");
    const out: Rec = {};
    for (const [k, x] of Object.entries(v as Rec)) if (x !== undefined) out[k] = plain(x);
    return out;
  }
  throw new TypeError(typeof v);
}

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

function contentOf(setting: Settings["logContent"]): boolean {
  if (setting !== undefined && setting !== null) return Boolean(setting);
  const raw = (env()["FUNCTAI_LOG_CONTENT"] ?? "").trim().toLowerCase();
  return !(OFF.has(raw) && raw !== "");
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
  signature?: string;
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
  response: Response | null;
  error: unknown;
  streamed: boolean;
  firstDelta: number | null;
}

/** One call being made. Only its id and timing exist when nothing is logged. */
export class Call {
  readonly id = newId();
  readonly parent: string | null;
  readonly root: string;
  readonly started = now();
  readonly exchanges: Exchange[] = [];
  provider: string | null = null;
  inputs: Rec | null = null;
  sizes: Record<string, number> = {};
  outputs: Rec | null = null;
  confidence: number | null = null;

  readonly program: () => Program;
  readonly folder: string | null;
  readonly content: boolean;
  readonly caller: Rec;

  constructor(program: () => Program, parent: Call | undefined, folder: string | null, content: boolean, caller: Rec) {
    this.program = program;
    this.folder = folder;
    this.content = content;
    this.caller = caller;
    this.parent = parent?.id ?? null;
    this.root = parent?.root ?? this.id;
  }

  exchange(model: string, request: Request, response: Response | null, started: number, seconds: number,
    opts: { cached?: boolean; error?: unknown; streamed?: boolean; firstDelta?: number | null } = {}): void {
    this.exchanges.push({
      model, provider: this.provider, started, seconds, cached: opts.cached ?? false, request, response,
      error: opts.error ?? null, streamed: opts.streamed ?? false, firstDelta: opts.firstDelta ?? null,
    });
  }
}

export const current = new Context<Call>();

/** Start a call of a program: an id, a parent, and a line in the log when logging is on. */
export function start(program: () => Program, settings: Settings, inputs: Rec): Call {
  let folder: string | null = null;
  let content = true;
  try {
    folder = folderOf(settings.logCalls);
    content = contentOf(settings.logContent);
  } catch (err) {
    warnOnce(`start:${(err as Error).name}`, `calls are not logged: ${(err as Error).message}`);
  }
  const call = new Call(program, current.get(), folder, content, callerOf(settings));
  if (folder) {
    const sizes: Record<string, number> = {};
    const values: Rec = {};
    for (const [k, v] of Object.entries(inputs)) {
      const [data, n] = toJson(v);
      values[k] = data;
      sizes[k] = n;
    }
    call.sizes = sizes;
    if (content) call.inputs = values;
  }
  return call;
}

function errorJson(err: unknown, content: boolean): Rec {
  const e = err as { name?: string; code?: unknown; message?: string; constructor?: { name?: string } };
  const out: Rec = { type: e?.constructor?.name && e.constructor.name !== "Error" ? e.constructor.name : (e?.name ?? "Error") };
  if (lmcc.isRefusal(err) && typeof e.code === "string") out["code"] = e.code;
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

/** The call's record (contract/calls.md, "A call record"). */
export function record(call: Call, opts: { returned?: unknown; error?: unknown; hasReturned?: boolean } = {}): Rec {
  const program = call.program();
  const content = call.content;
  let outputs: Rec | null = null;
  const outSizes: Record<string, number> = {};
  if (call.outputs) {
    outputs = {};
    for (const [k, v] of Object.entries(call.outputs)) {
      const [data, n] = toJson(v);
      outputs[k] = data;
      outSizes[k] = n;
    }
  }
  if (program.kind === "module" && opts.error === undefined && opts.hasReturned) {
    const [data, n] = toJson(opts.returned);
    outputs = { result: data };
    outSizes["result"] = n;
  }
  const answered = call.exchanges.filter((e) => e.response !== null);
  const usage: Record<string, number> = {};
  for (const e of answered) for (const [k, v] of Object.entries(usageOf(e.response!))) usage[k] = (usage[k] ?? 0) + v;
  const rec: Rec = {
    functai_call: FORMAT, id: call.id, parent: call.parent, root: call.root, program,
    started: iso(call.started), seconds: round6((now() - call.started) / 1000), content,
  };
  if (content) {
    rec["inputs"] = call.inputs ?? {};
    rec["outputs"] = outputs;
    if (program.kind === "ai" && opts.hasReturned && opts.error === undefined) {
      const [shown] = toJson(opts.returned);
      if (outputs === null || !lmcc.jsonEqual((outputs as Rec)[program.answer] as lmcc.Json, shown as lmcc.Json)) rec["returned"] = shown;
    }
  }
  rec["sizes"] = { inputs: call.sizes, outputs: outSizes };
  rec["error"] = opts.error !== undefined ? errorJson(opts.error, content) : null;
  rec["model"] = answered.length ? answered[answered.length - 1]!.model : null;
  rec["usage"] = usage;
  rec["confidence"] = call.confidence;
  rec["exchanges"] = call.exchanges.map((e) => exchangeJson(e, content));
  rec["caller"] = { ...call.caller };
  rec["process"] = processJson();
  return rec;
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
export function finish(call: Call, opts: { returned?: unknown; error?: unknown; hasReturned?: boolean }): void {
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
        if (r["functai_call"] === FORMAT && String(r["started"] ?? "") >= cutoff) calls.push(r);
        else if (r["functai_rating"] === FORMAT && String(r["at"] ?? "") >= cutoff) ratings.push(r);
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

function says(rating: Rec, call: Rec): Rec | null {
  const answer = ((call["program"] ?? {}) as Rec)["answer"] as string || "result";
  if (rating["verdict"] === "right") {
    const outputs = (call["outputs"] ?? {}) as Rec | null;
    return outputs && answer in outputs ? { [answer]: outputs[answer] } : null;
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
export function ratedRows(calls: Iterable<Rec>, ratings: Iterable<Rec>, opts: { name: string; module?: string | null; signature?: string | null; by?: string | null }): [Rec[], LeftOut] {
  const counting = currentRatings(ratings, opts.by);
  const left: LeftOut = { other_signature: 0, no_content: 0, no_answer: 0 };
  const rows: Rec[] = [];
  const mine = [...calls].filter((c) => {
    const p = (c["program"] ?? {}) as Rec;
    return p["name"] === opts.name && (opts.module === undefined || opts.module === null || p["module"] === opts.module);
  });
  mine.sort((a, b) => (order(a, "started") < order(b, "started") ? -1 : order(a, "started") > order(b, "started") ? 1 : 0));
  for (const call of mine) {
    const rs = counting.get(call["id"] as string);
    if (!rs || !rs.length) continue;
    const program = (call["program"] ?? {}) as Rec;
    if (opts.signature !== undefined && opts.signature !== null && program["signature"] !== opts.signature) {
      left.other_signature++;
      continue;
    }
    if (!call["content"] || !("inputs" in call)) {
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
  const rec: Rec = { functai_rating: FORMAT, id: newId(), call: callId, at: iso(Date.now()), by: who, verdict: v };
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
