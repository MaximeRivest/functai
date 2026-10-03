/**
 * Where conversations are kept (contract/conversations.md, *Stores*).
 *
 * A store keeps each conversation as an ordered list of records (JSON
 * objects) and never changes one. Every store has two methods, sync or async:
 *
 * - `append(conversation, records, { expect })`: add the records at the end,
 *   all or none, numbering them (`seq`: 1 for a conversation's first record);
 *   with `expect`, only when the conversation holds exactly that many now,
 *   else `ConversationError` (`store-conflict`). Returns how many it holds after.
 * - `read(conversation, after)`: the records after position `after`, in order.
 *
 * and may have `events` (a store of call tree logs, so another process can
 * watch a turn while it runs), `wait(conversation, after, ms)`,
 * `durability` (`"memory"`, `"disk"`) and `persistent` (whether it keeps
 * records beyond this process; unknown: yes).
 *
 * Two stores are here: `MemoryConversations` (the default: this process's
 * memory) and `FolderStore` (files in a folder, the same files Python, R and
 * Julia write, locked across processes).
 */

import { appendRules, finishedLog, MemoryStore, positionOf, resume, type AppendAnswer, type ClaimAnswer, type EventStore,
  type Position, type ReadAnswer } from "./events.ts";
import { ConversationError } from "./errors.ts";
import { builtin, env } from "./host.ts";
import { warnOnce } from "./calllog.ts";
import { copyData, parseData, writeData } from "./values.ts";

type Rec = Record<string, unknown>;

/** A conversation store: append and read, sync or async (a `Map`-backed one, a database, …). */
export interface ConversationStore {
  append(conversation: string, records: readonly Rec[], opts?: { expect?: number }): number | Promise<number>;
  read(conversation: string, after?: number): readonly Rec[] | Promise<readonly Rec[]>;
  /** Resolves when a record after `after` exists (or `ms` passed). */
  wait?(conversation: string, after: number, ms: number): void | Promise<void>;
  /** Each turn's call tree log, kept while it runs (streaming.md, *The rules a store keeps*). */
  readonly events?: EventStore | null;
  readonly durability?: string;
  /** Whether it keeps records beyond this process (unknown: yes). */
  readonly persistent?: boolean;
}

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/;

/** A conversation's id: 1 to 200 ASCII letters, digits, `.`, `_` or `-`, starting with a letter or digit (a file name everywhere, never a path). */
export function checkId(conversation: unknown): string {
  if (typeof conversation !== "string" || !ID.test(conversation)) {
    throw new ConversationError("conversation-id", `a conversation's id is 1 to 200 letters, digits, '.', '_' or '-', starting with a letter or digit; not ${JSON.stringify(conversation)}`);
  }
  return conversation;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ memory

/**
 * Conversations kept in this process's memory (lost when it ends): the store
 * a conversation uses when none is named. One per process by default, so
 * opening the same id again opens the same conversation.
 */
export class MemoryConversations implements ConversationStore {
  readonly durability = "memory";
  readonly persistent = false;
  readonly events = new MemoryStore();
  private readonly records = new Map<string, Rec[]>();
  private waiters: Array<() => void> = [];

  append(conversation: string, records: readonly Rec[], opts: { expect?: number } = {}): number {
    checkId(conversation);
    const log = this.records.get(conversation) ?? [];
    if (opts.expect !== undefined && opts.expect !== log.length) {
      throw new ConversationError("store-conflict", `conversation ${conversation} holds ${log.length} records, not ${opts.expect}`);
    }
    for (const r of records) log.push({ ...copyData(r), seq: log.length + 1 });
    this.records.set(conversation, log);
    const w = this.waiters;
    this.waiters = [];
    for (const f of w) f();
    return log.length;
  }

  read(conversation: string, after = 0): Rec[] {
    return copyData((this.records.get(conversation) ?? []).slice(after));
  }

  async wait(conversation: string, after: number, ms: number): Promise<void> {
    if ((this.records.get(conversation)?.length ?? 0) > after) return;
    await new Promise<void>((resolve) => {
      const t = setTimeout(done, ms);
      function done() {
        clearTimeout(t);
        resolve();
      }
      this.waiters.push(done);
    });
  }

  /** The conversations it holds. */
  conversations(): string[] {
    return [...this.records.keys()].filter((c) => this.records.get(c)!.length);
  }
}

/** This process's memory store (`store: null`). */
export const MEMORY = new MemoryConversations();

// ------------------------------------------------------------------ files

/**
 * Hold an exclusive lock on a file across processes while `fn` runs: the
 * same `flock` Python's and Julia's `FolderStore` take on the same file, so
 * every language's writers exclude each other. Node has no `flock` call of its
 * own; the open file is handed to the system's `flock` command (util-linux,
 * as lm15 does), which locks it for this process. Where there is none (most
 * Macs, Windows), a lock directory beside the file excludes this package's
 * own writers only, with one warning.
 */
const inProcess = new Map<string, Promise<void>>();

async function flockOf(fd: number): Promise<boolean | null> {
  const cp = builtin("node:child_process" as never) as typeof import("node:child_process") | null;
  if (!cp) return null;
  return new Promise((resolve) => {
    let child: import("node:child_process").ChildProcess;
    try {
      child = cp.spawn("flock", ["--exclusive", "--nonblock", "--conflict-exit-code", "75", "3"],
        { stdio: ["ignore", "ignore", "ignore", fd], timeout: 5000 });
    } catch {
      resolve(null);
      return;
    }
    child.once("error", () => resolve(null));
    child.once("close", (code) => resolve(code === 0 ? true : code === 75 ? false : null));
  });
}

let noFlock = false;

export async function withLock<T>(path: string, fn: () => T | Promise<T>): Promise<T> {
  const fs = builtin("node:fs")!;
  // one holder at a time in this process (flock would exclude it too: this saves the polling)
  for (;;) {
    const busy = inProcess.get(path);
    if (!busy) break;
    await busy;
  }
  let release!: () => void;
  inProcess.set(path, new Promise<void>((r) => { release = r; }));
  const fd = fs.openSync(path, fs.constants.O_RDWR | fs.constants.O_CREAT, 0o600);
  let dir: string | null = null;
  try {
    const deadline = Date.now() + 60_000;
    let pause = 5;
    for (;;) {
      const got = noFlock ? null : await flockOf(fd);
      if (got === true) break;
      if (got === null) {
        noFlock = true;
        warnOnce("no-flock", "this system has no `flock` command: a conversation folder is locked against this package's own processes only, not Python's, R's or Julia's");
        dir = await lockDir(path + ".d", deadline);
        break;
      }
      if (Date.now() > deadline) throw new Error(`could not lock ${path} in 60 seconds`);
      await sleep(pause);
      pause = Math.min(pause * 2, 100);
    }
    return await fn();
  } finally {
    fs.closeSync(fd);                                   // closing releases flock
    if (dir) fs.rmSync(dir, { recursive: true, force: true });
    inProcess.delete(path);
    release();
  }
}

async function lockDir(dir: string, deadline: number): Promise<string> {
  const fs = builtin("node:fs")!;
  for (;;) {
    try {
      fs.mkdirSync(dir, { mode: 0o700 });
      return dir;
    } catch {
      try {
        if (Date.now() - fs.statSync(dir).mtimeMs > 30_000) fs.rmSync(dir, { recursive: true, force: true });   // a holder that died
      } catch {
        // gone meanwhile
      }
      if (Date.now() > deadline) throw new Error(`could not lock ${dir} in 60 seconds`);
      await sleep(20);
    }
  }
}

/** A JSON lines file read incrementally: what was parsed is kept, and only what was appended since is read again. */
class Lines {
  private size = 0;
  items: Rec[] = [];
  readonly path: string;
  constructor(path: string) {
    this.path = path;
  }

  load(): Rec[] {
    const fs = builtin("node:fs")!;
    let size: number;
    try {
      size = fs.statSync(this.path).size;
    } catch {
      this.size = 0;
      this.items = [];
      return this.items;
    }
    if (size < this.size) {                            // replaced: read it all again
      this.size = 0;
      this.items = [];
    }
    if (size > this.size) {
      const fd = fs.openSync(this.path, "r");
      const buf = Buffer.alloc(size - this.size);
      try {
        fs.readSync(fd, buf, 0, buf.length, this.size);
      } finally {
        fs.closeSync(fd);
      }
      const end = buf.lastIndexOf(10) + 1;             // a line still being written is read next time
      for (const raw of buf.subarray(0, end).toString("utf8").split("\n")) {
        if (!raw.trim()) continue;
        try {
          this.items.push(parseData(raw) as Rec);
        } catch {
          this.items.push({ unreadable: true });
        }
      }
      this.size += end;
    }
    return this.items;
  }
}

function write(path: string, lines: readonly string[], durable: boolean): void {
  const fs = builtin("node:fs")!;
  const fd = fs.openSync(path, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_APPEND, 0o600);
  try {
    fs.writeSync(fd, lines.join(""));
    if (durable) fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
}

const dump = (r: Rec) => writeData(r) + "\n";

function safe(name: unknown): string {
  if (typeof name !== "string" || !ID.test(name)) throw new TypeError(`${JSON.stringify(name)} is not a log's id`);
  return name;
}

/**
 * Call tree logs kept in files, by the rules every store keeps
 * (streaming.md): `<folder>/<tree>.jsonl` holds the kept events,
 * `<tree>.writer` the last writer number a claim gave, `<tree>.lock` is held
 * for each claim and append (the lock Python's takes).
 */
export class FolderEvents implements EventStore {
  readonly folder: string;
  private readonly files = new Map<string, Lines>();

  constructor(folder: string) {
    const fs = builtin("node:fs")!;
    fs.mkdirSync(folder, { recursive: true, mode: 0o700 });
    this.folder = folder;
  }

  private lines(tree: string): Lines {
    let f = this.files.get(tree);
    if (!f) this.files.set(tree, f = new Lines(builtin("node:path")!.join(this.folder, `${safe(tree)}.jsonl`)));
    return f;
  }

  private file(tree: string, ext: string): string {
    return builtin("node:path")!.join(this.folder, `${safe(tree)}.${ext}`);
  }

  private writerOf(tree: string): number {
    try {
      return Number(builtin("node:fs")!.readFileSync(this.file(tree, "writer"), "utf8").trim()) || 1;
    } catch {
      return 1;
    }
  }

  async append(events: readonly Readonly<Rec>[]): Promise<AppendAnswer> {
    if (!events.length) return "duplicate";
    const tree = events[0]!["tree"];
    if (typeof tree !== "string" || !ID.test(tree)) {
      const e = events[0]!;
      return { refuses: "event-malformed", event: { writer: Number(e["writer"]) || 1, seq: Number(e["seq"]) || 1 } };
    }
    return withLock(this.file(tree, "lock"), () => {
      const kept = this.lines(tree).load().filter((e) => !("unreadable" in e));
      const [answer, after] = appendRules(kept, this.writerOf(tree), events);
      if (after) write(this.file(tree, "jsonl"), after.slice(kept.length).map(dump), false);
      return answer;
    });
  }

  async claim(tree: string): Promise<ClaimAnswer> {
    const fs = builtin("node:fs")!;
    return withLock(this.file(tree, "lock"), () => {
      const log = this.lines(tree).load().filter((e) => !("unreadable" in e));
      if (!log.length) return { refuses: "event-unknown" as const };
      if (finishedLog(log, tree)) return { refuses: "event-after-end" as const };
      const writer = this.writerOf(tree) + 1;
      const tmp = this.file(tree, "writer.tmp");
      fs.writeFileSync(tmp, String(writer), { mode: 0o600 });
      fs.renameSync(tmp, this.file(tree, "writer"));
      return { writer, after: positionOf(log[log.length - 1]!) };
    });
  }

  async read(tree: string, after: Position | null): Promise<ReadAnswer> {
    return copyData(resume(this.lines(tree).load().filter((e) => !("unreadable" in e)), after) as ReadAnswer);
  }
}

/** Where `store: true` keeps conversations: `$XDG_DATA_HOME/functai/conversations` (beside the call log's folder). */
export function defaultConversationsFolder(): string | null {
  const os = builtin("node:os");
  const path = builtin("node:path");
  if (!os || !path) return null;
  const e = env();
  let base: string;
  if (os.platform() === "darwin") base = path.join(os.homedir(), "Library", "Application Support");
  else if (os.platform() === "win32") base = e["LOCALAPPDATA"] ?? path.join(os.homedir(), "AppData", "Local");
  else base = e["XDG_DATA_HOME"] || path.join(os.homedir(), ".local", "share");
  return path.join(base, "functai", "conversations");
}

/**
 * Conversations kept in a folder, shared by every process that opens it, in
 * any language:
 *
 * ```
 * <folder>/conversations/<id>.jsonl   the records, one per line
 * <folder>/conversations/<id>.lock    taken while appending
 * <folder>/trees/<tree>.jsonl         each turn's call tree log (kept form)
 * ```
 *
 * Each append is one step across processes (a lock on the conversation),
 * written and flushed to disk before it returns (`durability` `"disk"`).
 * Files are readable by their owner only.
 */
export class FolderStore implements ConversationStore {
  readonly durability = "disk";
  readonly persistent = true;
  readonly folder: string;
  readonly events: FolderEvents;
  private readonly dir: string;
  private readonly files = new Map<string, Lines>();

  constructor(folder: string) {
    const fs = builtin("node:fs");
    const path = builtin("node:path");
    const os = builtin("node:os");
    if (!fs || !path) throw new Error("a folder store needs a file system; keep conversations in memory (store: null) or give your own store");
    const home = folder === "~" || folder.startsWith("~/") ? (os?.homedir() ?? "~") + folder.slice(1) : folder;
    this.folder = path.resolve(home);
    this.dir = path.join(this.folder, "conversations");
    fs.mkdirSync(this.dir, { recursive: true, mode: 0o700 });
    this.events = new FolderEvents(path.join(this.folder, "trees"));
  }

  private lines(conversation: string): Lines {
    let f = this.files.get(conversation);
    if (!f) this.files.set(conversation, f = new Lines(builtin("node:path")!.join(this.dir, `${conversation}.jsonl`)));
    return f;
  }

  async append(conversation: string, records: readonly Rec[], opts: { expect?: number } = {}): Promise<number> {
    checkId(conversation);
    const path = builtin("node:path")!;
    return withLock(path.join(this.dir, `${conversation}.lock`), () => {
      let n = this.lines(conversation).load().length;
      if (opts.expect !== undefined && opts.expect !== n) {
        throw new ConversationError("store-conflict", `conversation ${conversation} holds ${n} records, not ${opts.expect}`);
      }
      const lines = records.map((r) => dump({ ...r, seq: ++n }));
      if (lines.length) write(path.join(this.dir, `${conversation}.jsonl`), lines, true);
      return n;
    });
  }

  read(conversation: string, after = 0): Rec[] {
    checkId(conversation);
    return copyData(this.lines(conversation).load().slice(after));
  }

  async wait(conversation: string, after: number, ms: number): Promise<void> {
    const deadline = Date.now() + ms;
    let pause = 20;
    while (Date.now() < deadline) {
      if (this.lines(conversation).load().length > after) return;
      await sleep(Math.min(pause, Math.max(0, deadline - Date.now())));
      pause = Math.min(pause * 2, 200);
    }
  }

  /** The conversations it holds. */
  conversations(): string[] {
    const fs = builtin("node:fs")!;
    return fs.readdirSync(this.dir).filter((f) => f.endsWith(".jsonl")).map((f) => f.slice(0, -6)).sort();
  }
}

const folders = new Map<string, FolderStore>();

/**
 * The store a `store` value names: null (this process's memory), true (the
 * default folder), a folder, or an object with `append` and `read`.
 */
export function storeOf(store: unknown): ConversationStore {
  if (store === undefined || store === null || store === false) return MEMORY;
  if (store === true) {
    const where = defaultConversationsFolder();
    if (!where) throw new Error("this runtime has no file system: keep conversations in memory (store: null) or give your own store");
    store = where;
  }
  if (typeof store === "string") {
    const path = builtin("node:path");
    const os = builtin("node:os");
    const where = path ? path.resolve(store === "~" || store.startsWith("~/") ? (os?.homedir() ?? "~") + store.slice(1) : store) : store;
    let found = folders.get(where);
    if (!found) folders.set(where, found = new FolderStore(where));
    return found;
  }
  if (typeof store === "object" && typeof (store as ConversationStore).append === "function" && typeof (store as ConversationStore).read === "function") {
    return store as ConversationStore;
  }
  throw new TypeError("store is null (memory), true (the default folder), a folder, or an object with append(conversation, records, { expect }) and read(conversation, after)");
}

/** Whether a store keeps records beyond this process (unknown: yes). */
export function isPersistent(store: ConversationStore): boolean {
  return store.persistent === undefined ? true : Boolean(store.persistent);
}

/** Wait for a record after `after` (at most `ms`). */
export async function waitFor(store: ConversationStore, conversation: string, after: number, ms: number): Promise<void> {
  if (store.wait) await store.wait(conversation, after, ms);
  else await sleep(Math.min(ms, 100));
}

/**
 * The kept events of a tree's log from a store, after `after`: those kept so
 * far, then each as it is kept, until the log's last event (or `stop()` says
 * so, or `timeout` ms pass without one).
 */
export async function* follow(store: ConversationStore, tree: string, after: Position | null = null,
  opts: { poll?: number; stop?: () => boolean | Promise<boolean>; timeout?: number; signal?: AbortSignal } = {}): AsyncGenerator<Rec> {
  const events = store.events;
  if (!events) return;
  let last = after;
  let quiet = Date.now();
  for (;;) {
    if (opts.signal?.aborted) return;
    const got = await events.read(tree, last);
    if ("refuses" in got) return;
    for (const e of got.events as unknown as Rec[]) {
      last = positionOf(e);
      quiet = Date.now();
      yield e;
      if (e["call"] === tree && (e["kind"] === "done" || e["kind"] === "failed")) return;
    }
    if (opts.stop && await opts.stop()) return;
    if (opts.timeout !== undefined && Date.now() - quiet > opts.timeout) return;
    await sleep(opts.poll ?? 50);
  }
}
