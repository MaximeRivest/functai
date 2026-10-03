/**
 * The reply cache (contract/replies.md): a model's reply to a request, reused
 * when the same request is made again.
 *
 * ```ts
 * configure({ cacheReplies: "disk" });      // kept across runs, in the SQLite file every language shares
 * configure({ cacheReplies: true });        // in this process's memory only
 * fn.using({ replicate: 2 });               // a third, independent answer to the same request
 * ```
 *
 * A reply is kept only once it was read (lmcc read it, and its values fit
 * their types), so an interrupted run leaves nothing half-written. One flight
 * per key: while one caller asks the model, another caller of the same
 * request waits for its reply, in this process and across processes (a claim
 * with a lease, in the same SQLite file). A call whose `logContent` drops a
 * field is never written to a store that outlives the process: it holds whole
 * requests and replies, which is what `logContent` forbids keeping.
 */

import { Request, Response, parseJson } from "@lm15/lm15";
import * as lmcc from "lmcc";
import * as bridge from "lmcc/lm15";
import { iso, warnOnce } from "./calllog.ts";
import { Cancelled } from "./errors.ts";
import { builtin, env, pid } from "./host.ts";

type Rec = Record<string, unknown>;

/** The reply cache's format (replies.md). */
export const FORMAT = 1;
/** Seconds a claim holds a key against other processes. */
const LEASE = 120;

/**
 * Where replies are kept, when it is your own: a `Map` works, or any store
 * with `get` and `set` (or `put`), sync or async (Redis, a KV store, …). It is
 * given plain JSON under string keys. `delete` (or `discard`) forgets a kept
 * reply that no longer reads; `claim(key)` and `unclaim(key)` make one flight
 * per key across the processes that share it. `durable: false` says it keeps
 * nothing beyond the process (then a call whose `logContent` drops a field
 * may use it).
 */
export interface ReplyCache {
  get(key: string): unknown;
  set?(key: string, reply: Rec): unknown;
  put?(key: string, reply: Rec): unknown;
  delete?(key: string): unknown;
  discard?(key: string): unknown;
  claim?(key: string, signal?: AbortSignal): unknown;
  unclaim?(key: string): unknown;
  readonly durable?: boolean;
}

/** What `cacheReplies` takes: off, this process's memory, the shared file, a folder or `.sqlite` path, or your own store. */
export type CacheSetting = boolean | "memory" | "disk" | string | ReplyCache | null;

/**
 * The cache key of a request (replies.md, "The key"): `sha256:` of the
 * canonical JSON of `{"functai_reply": 1, "request": <lm15's canonical
 * request>, "replicate": n}`, the same in every language.
 */
export function replyKey(request: Request, replicate = 0): string {
  return lmcc.sha256({ functai_reply: FORMAT, request: Request.toJSON(request), replicate: Math.trunc(replicate || 0) } as unknown as lmcc.Json);
}

const sleep = (ms: number, signal?: AbortSignal) => new Promise<void>((resolve, reject) => {
  if (signal?.aborted) return reject(new Cancelled());
  const t = setTimeout(() => { signal?.removeEventListener("abort", stop); resolve(); }, ms);
  const stop = () => { clearTimeout(t); reject(new Cancelled()); };
  signal?.addEventListener("abort", stop, { once: true });
});

/** One flight per key in this process: who holds each key, and what the next waits for. */
class Keys {
  private readonly held = new Map<string, Promise<void>>();

  async acquire(key: string, signal?: AbortSignal): Promise<void> {
    for (;;) {
      const busy = this.held.get(key);
      if (!busy) break;
      await new Promise<void>((resolve, reject) => {
        if (signal?.aborted) return reject(new Cancelled());
        const stop = () => reject(new Cancelled());
        signal?.addEventListener("abort", stop, { once: true });
        busy.then(() => { signal?.removeEventListener("abort", stop); resolve(); });
      });
    }
    let release!: () => void;
    this.held.set(key, new Promise<void>((r) => { release = r; }));
    (this.releasers ??= new Map()).set(key, release);
  }

  private releasers?: Map<string, () => void>;

  release(key: string): void {
    const r = this.releasers?.get(key);
    this.releasers?.delete(key);
    this.held.delete(key);
    r?.();
  }
}

/** Replies kept in this process's memory, at most `capacity` (the least recently used go first). */
export class MemoryReplies implements ReplyCache {
  readonly durable = false;
  private readonly data = new Map<string, Rec>();
  private readonly keys = new Keys();
  private readonly capacity: number;
  constructor(capacity = 20_000) {
    this.capacity = capacity;
  }

  get(key: string): Rec | undefined {
    const hit = this.data.get(key);
    if (hit !== undefined) {
      this.data.delete(key);
      this.data.set(key, hit);
    }
    return hit;
  }

  set(key: string, reply: Rec): void {
    this.data.delete(key);
    this.data.set(key, reply);
    while (this.data.size > this.capacity) this.data.delete(this.data.keys().next().value!);
  }

  delete(key: string): void {
    this.data.delete(key);
  }

  clear(): void {
    this.data.clear();
  }

  get size(): number {
    return this.data.size;
  }

  async claim(key: string, signal?: AbortSignal): Promise<Rec | undefined> {
    await this.keys.acquire(key, signal);
    return this.get(key);
  }

  unclaim(key: string): void {
    this.keys.release(key);
  }
}

/** The file `cacheReplies: "disk"` keeps replies in: the user's cache folder (replies.md, "The file"). */
export function defaultCachePath(): string | null {
  const os = builtin("node:os");
  const path = builtin("node:path");
  if (!os || !path) return null;
  const e = env();
  let base: string;
  if (os.platform() === "darwin") base = path.join(os.homedir(), "Library", "Caches");
  else if (os.platform() === "win32") base = e["LOCALAPPDATA"] || path.join(os.homedir(), "AppData", "Local");
  else base = e["XDG_CACHE_HOME"] || path.join(os.homedir(), ".cache");
  return path.join(base, "functai", "replies.sqlite");
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, format INTEGER NOT NULL, created TEXT NOT NULL, model TEXT, response TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS claims (key TEXT PRIMARY KEY, owner TEXT NOT NULL, until REAL NOT NULL);
`;

type Database = {
  exec(sql: string): void;
  prepare(sql: string): { get(...a: unknown[]): Rec | undefined; run(...a: unknown[]): unknown };
  close(): void;
};

let owner: string | null = null;
/** `<host>:<pid>:<random>`: who holds a claim. */
export function holderName(): string {
  if (!owner) {
    const os = builtin("node:os");
    const rand = Array.from(globalThis.crypto.getRandomValues(new Uint8Array(3)), (b) => b.toString(16).padStart(2, "0")).join("");
    owner = `${os ? os.hostname() : "host"}:${pid() ?? 0}:${rand}`;
  }
  return owner;
}

function expandPath(p: string): string {
  const os = builtin("node:os");
  const path = builtin("node:path")!;
  return path.resolve(p === "~" || p.startsWith("~/") ? (os?.homedir() ?? "~") + p.slice(1) : p);
}

/**
 * Replies kept in one SQLite file (replies.md, "The file"), shared by every
 * process that opens it, whatever its language: writes are transactions, so
 * nothing is half-written; a claim with a lease makes one flight per key
 * across processes. Created readable by its owner only. Needs `node:sqlite`
 * (Node 22.13 or later, Deno).
 */
export class DiskReplies implements ReplyCache {
  readonly durable = true;
  readonly path: string;
  private readonly db: Database;
  private readonly keys = new Keys();
  private readonly lease: number;

  constructor(path?: string | null, lease = LEASE) {
    this.lease = lease;
    const sqlite = builtin("node:sqlite" as never) as { DatabaseSync: new (p: string) => Database } | null;
    const nodePath = builtin("node:path");
    const fs = builtin("node:fs");
    if (!sqlite || !nodePath || !fs) throw new Error("a reply cache on disk needs node:sqlite (Node 22.13 or later); use cacheReplies: true (memory) here");
    let p = path ? expandPath(path) : defaultCachePath();
    if (!p) throw new Error("no cache folder on this system");
    if (![".sqlite", ".sqlite3", ".db"].some((s) => p!.endsWith(s))) p = nodePath.join(p, "replies.sqlite");
    this.path = p;
    fs.mkdirSync(nodePath.dirname(p), { recursive: true, mode: 0o700 });
    const fresh = !fs.existsSync(p);
    this.db = new sqlite.DatabaseSync(p);
    this.db.exec("PRAGMA busy_timeout = 30000");
    this.db.exec("PRAGMA journal_mode = WAL");
    this.db.exec("PRAGMA synchronous = NORMAL");
    this.db.exec(SCHEMA);
    if (fresh) {
      try {
        fs.chmodSync(p, 0o600);
      } catch {
        // a file system without modes
      }
    }
  }

  get(key: string): Rec | undefined {
    const row = this.db.prepare("SELECT response, format FROM replies WHERE key = ?").get(key);
    if (!row || row["format"] !== FORMAT) return undefined;
    try {
      return parseJson(row["response"] as string) as Rec;      // lm15 reads it: member order and big integers kept
    } catch {
      return undefined;                                         // a row this reader cannot read is no reply
    }
  }

  set(key: string, reply: Rec): void {
    const model = typeof reply["model"] === "string" ? reply["model"] : null;
    this.transaction(() => {
      this.db.prepare("INSERT OR REPLACE INTO replies (key, format, created, model, response) VALUES (?, ?, ?, ?, ?)")
        .run(key, FORMAT, iso(Date.now()), model, lmcc.canonicalJson(reply));
      this.db.prepare("DELETE FROM claims WHERE key = ? AND owner = ?").run(key, holderName());
    });
  }

  delete(key: string): void {
    this.db.prepare("DELETE FROM replies WHERE key = ?").run(key);
  }

  clear(): void {
    this.db.exec("DELETE FROM replies; DELETE FROM claims;");
  }

  get size(): number {
    return Number(this.db.prepare("SELECT COUNT(*) AS n FROM replies").get()!["n"]);
  }

  private transaction<T>(fn: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const out = fn();
      this.db.exec("COMMIT");
      return out;
    } catch (err) {
      this.db.exec("ROLLBACK");
      throw err;
    }
  }

  /**
   * The reply to `key` when one is kept; else take the key (one flight):
   * wait while another call of this process or another process holds it.
   */
  async claim(key: string, signal?: AbortSignal): Promise<Rec | undefined> {
    await this.keys.acquire(key, signal);
    let pause = 50;
    try {
      for (;;) {
        const hit = this.get(key);
        if (hit) return hit;
        const now = Date.now() / 1000;
        const mine = this.transaction(() => {
          const row = this.db.prepare("SELECT owner, until FROM claims WHERE key = ?").get(key);
          const free = !row || row["owner"] === holderName() || (row["until"] as number) < now;
          if (free) this.db.prepare("INSERT OR REPLACE INTO claims (key, owner, until) VALUES (?, ?, ?)").run(key, holderName(), now + this.lease);
          return free;
        });
        if (mine) return undefined;
        await sleep(pause, signal);
        pause = Math.min(pause * 2, 500);
      }
    } catch (err) {
      this.keys.release(key);
      throw err;
    }
  }

  unclaim(key: string): void {
    try {
      this.db.prepare("DELETE FROM claims WHERE key = ? AND owner = ?").run(key, holderName());
    } finally {
      this.keys.release(key);
    }
  }

  close(): void {
    this.db.close();
  }
}

const memory = new MemoryReplies();
const disks = new Map<string, DiskReplies>();

/** The store a `cacheReplies` setting names, or null when replies are not cached. */
export function cacheOf(setting: unknown): ReplyCache | null {
  if (setting === undefined || setting === null || setting === false) return null;
  if (setting === true || setting === "memory") return memory;
  if (typeof setting === "string") {
    const where = setting === "disk" ? defaultCachePath() : expandPath(setting);
    if (!where) throw new Error("this runtime has no file system: cacheReplies: \"disk\" cannot be used (use true: memory)");
    let store = disks.get(where);
    if (!store) disks.set(where, store = new DiskReplies(setting === "disk" ? null : setting));
    return store;
  }
  if (typeof setting === "object" && typeof (setting as ReplyCache).get === "function"
    && (typeof (setting as ReplyCache).set === "function" || typeof (setting as ReplyCache).put === "function")) return setting as ReplyCache;
  throw new TypeError("cacheReplies is false, true (memory), \"disk\", a folder or .sqlite path, or a store with get and set (a Map works)");
}

/** Refuse, where it is set, a `cacheReplies` value that could only fail later. */
export function checkCacheSetting(setting: unknown, where: string): void {
  if (setting === undefined || setting === null || typeof setting === "boolean") return;
  if (typeof setting === "string") {
    if (!setting.trim()) throw new TypeError(`${where}: cacheReplies is false, true, "disk" or a path, not an empty text`);
    return;
  }
  if (typeof setting === "object" && typeof (setting as ReplyCache).get === "function"
    && (typeof (setting as ReplyCache).set === "function" || typeof (setting as ReplyCache).put === "function")) return;
  throw new TypeError(`${where}: cacheReplies is false, true (memory), "disk", a folder or .sqlite path, or a store with get and set (a Map works)`);
}

/**
 * Forget kept replies: this process's memory (default), or the store a
 * `cacheReplies` value names (`"disk"`, a path).
 */
export function clearCache(which: CacheSetting = null): void {
  if (which === null || which === true || which === "memory") {
    memory.clear();
    return;
  }
  const store = cacheOf(which) as { clear?: () => void } | null;
  store?.clear?.();
}

// A store that fails is not a reason to fail the call: it is skipped, with one warning.
const skip = (what: string, err: unknown) => warnOnce(`cache:${what}`, `the reply cache could not ${what} (${(err as Error)?.message ?? err}); calling the model instead`);

function asResponse(hit: unknown): Response | null {
  if (typeof hit === "string") hit = parseJson(hit);     // a store that keeps text: lm15 reads it, member order and big integers kept
  return hit && typeof hit === "object" ? Response.fromJSON(bridge.toLm15(hit) as never) : null;
}

/**
 * One request's turn at the cache: `reply` (a kept reply, or null: this
 * caller asks the model), then `keep(response)` once the reply was read, or
 * `drop()` when it could not be (a kept reply that no longer reads is
 * forgotten); `end()` always.
 */
export class Flight {
  private ended = false;
  readonly store: ReplyCache;
  readonly key: string;
  readonly reply: Response | null;
  constructor(store: ReplyCache, key: string, reply: Response | null) {
    this.store = store;
    this.key = key;
    this.reply = reply;
  }

  async keep(response: Response): Promise<void> {
    if (this.reply !== null) return;
    try {
      const json = Response.toJSON(response) as Rec;
      await (this.store.set ? this.store.set(this.key, json) : this.store.put!(this.key, json));
    } catch (err) {
      skip("write", err);
    }
  }

  async drop(): Promise<void> {
    if (this.reply === null) return;
    try {
      const forget = this.store.delete ?? this.store.discard;
      await forget?.call(this.store, this.key);
    } catch (err) {
      skip("forget", err);
    }
  }

  async end(): Promise<void> {
    if (this.ended) return;
    this.ended = true;
    try {
      await this.store.unclaim?.(this.key);
    } catch (err) {
      skip("release", err);
    }
  }
}

/**
 * The cache's turn for this request under this setting, or null when no
 * cache applies. A store that keeps replies beyond the process is skipped
 * (memory is used instead) for a call whose `logContent` drops a field.
 */
export async function begin(setting: unknown, request: Request, replicate: number, contentWhole: boolean, signal?: AbortSignal): Promise<Flight | null> {
  let store: ReplyCache | null;
  try {
    store = cacheOf(setting);
  } catch (err) {
    warnOnce(`replies-store:${(err as Error).name}`, `the reply cache cannot be opened (${(err as Error).message}); replies are not cached`);
    return null;
  }
  if (!store) return null;
  if (store.durable !== false && !contentWhole) store = memory;
  const key = replyKey(request, replicate);
  let hit: unknown;
  try {
    hit = store.claim ? await store.claim(key, signal) : await store.get(key);
  } catch (err) {
    if (err instanceof Cancelled) throw err;
    skip("read", err);
    return null;
  }
  let reply: Response | null = null;
  try {
    reply = asResponse(hit);
  } catch (err) {
    skip("read", err);
  }
  return new Flight(store, key, reply);
}
