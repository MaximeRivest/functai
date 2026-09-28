/**
 * The writer of a call tree's log (contract/streaming.md): one log per tree,
 * numbered as its events happen, given to the streams opened in it, to
 * observers (the kept form) and to the tree's journal (the kept form, with
 * acknowledged conditional appends: best effort, or required, with barriers
 * at the tree's start, before each tool and at its end).
 */

import { iso, warnOnce } from "./calllog.ts";
import {
  keptEvent, positionOf, Relink, settle, type AppendAnswer, type EventStore, type KeptFields, type Position, type StreamEvent,
} from "./events.ts";

type Rec = Record<string, unknown>;

/** Gets the kept form of every event of every call in its scope, in order, as it happens. Never slows the call. */
export type Observer = (event: StreamEvent) => void | PromiseLike<void>;

/**
 * A journal: a store that keeps whole trees' kept logs while they are
 * written. A store alone is best effort; `{ store, mode: "required" }` makes
 * calls wait until their events are kept (at the tree's start, before each
 * tool, at its end).
 */
export type Journal = EventStore | JournalSetting;

export interface JournalSetting {
  readonly store: EventStore;
  /** `"best-effort"` (the default): the call never waits. `"required"`: it waits at three barriers, and raises `JournalError` when they fail. */
  readonly mode?: "required" | "best-effort";
  /** Sends of one append, after the first, before giving up on it for now (default 2). */
  readonly retries?: number;
  /** Most events in one append (default 64; 1 sends each event alone). */
  readonly batch?: number;
}

/** A journal setting as the rules read it: its store and mode (two are the same when both are). */
export interface ResolvedJournal {
  readonly store: EventStore;
  readonly mode: "required" | "best-effort";
  readonly retries: number;
  readonly batch: number;
}

const isStore = (x: unknown): x is EventStore => typeof x === "object" && x !== null && typeof (x as EventStore).append === "function";

/** A `journal` setting as its store and mode (null: no journal). */
export function journalOf(j: Journal | null): ResolvedJournal | null {
  if (j === null) return null;
  if (isStore(j)) return { store: j, mode: "best-effort", retries: 2, batch: 64 };
  if (typeof j !== "object" || !isStore(j.store)) throw new TypeError("journal: a store ({ append, read }), or { store, mode: \"required\" | \"best-effort\" }");
  const mode = j.mode ?? "best-effort";
  if (mode !== "required" && mode !== "best-effort") throw new TypeError(`journal mode is "required" or "best-effort", not ${JSON.stringify(mode)}`);
  return { store: j.store, mode, retries: Math.max(0, j.retries ?? 2), batch: Math.max(1, j.batch ?? 64) };
}

export type JournalCode = "journal-policy" | "journal-scope" | "journal-barrier" | "journal-end";

/** A call's outcome as `JournalError` holds it: its value, or its error. */
export type Outcome<A = unknown> = { readonly done: A } | { readonly failed: unknown };

/**
 * A journal decided a call (streaming.md, "Keeping a log while it is
 * written"). `journal-policy`: the layers around the tree break the journal
 * policy (refused before it runs). `journal-scope`: a required journal set
 * only inside a tree. `journal-barrier`: a required journal did not confirm
 * the events before the code or a tool ran. `journal-end`: it did not confirm
 * the call's end: `outcome` is what the call did (its value, or its error),
 * `event` names the event that records it, and `journal` says whether the
 * journal refused it or did not answer (`settle()` finds out which).
 */
export class JournalError<A = unknown> extends Error {
  readonly code: JournalCode;
  readonly journal?: "refused" | "unknown";
  readonly event?: Position;
  readonly outcome?: Outcome<A>;
  readonly tree?: string;
  private readonly store?: EventStore;

  constructor(code: JournalCode, message: string, opts: {
    journal?: "refused" | "unknown"; event?: Position; outcome?: Outcome<A>; tree?: string; store?: EventStore;
  } = {}) {
    super(message);
    this.name = "JournalError";
    this.code = code;
    if (opts.journal) this.journal = opts.journal;
    if (opts.event) this.event = opts.event;
    if (opts.outcome) this.outcome = opts.outcome;
    if (opts.tree) this.tree = opts.tree;
    Object.defineProperty(this, "store", { value: opts.store, enumerable: false });
  }

  /**
   * For `journal-end`: read the journal and say what became of the call's
   * end: `"kept"`, `"not-kept"` (the log is unfinished; final only once you
   * have claimed it) or `"another-end"` (another writer ended the log).
   */
  async settle(): Promise<"kept" | "not-kept" | "another-end"> {
    if (!this.store || !this.tree || !this.event) throw new Error("only a journal-end error can be settled");
    return settle(this.store, this.tree, this.event);
  }
}

type Status = "confirmed" | "refused" | "unanswered";

/**
 * Sends a tree's kept events to its journal, in order, each append naming
 * where it goes (`after`). It keeps every event not confirmed and sends it
 * again; after a refusal it sends nothing more.
 */
export class JournalWriter {
  private readonly pending: { e: Rec; n: number }[] = [];
  private added = 0;
  private confirmedN = 0;
  private active: Promise<Status> | null = null;
  refused = false;
  readonly journal: ResolvedJournal;
  private readonly onTrouble: (what: string) => void;

  constructor(journal: ResolvedJournal, onTrouble: (what: string) => void = () => undefined) {
    this.journal = journal;
    this.onTrouble = onTrouble;
  }

  /** Queue an event (in the kept form); returns its number in this writer, for `confirm`. */
  add(e: Rec): number {
    const n = ++this.added;
    if (this.refused) return n;
    this.pending.push({ e, n });
    this.kick();
    return n;
  }

  private kick(): Promise<Status> {
    this.active ??= this.run();
    return this.active;
  }

  private async run(): Promise<Status> {
    const { store, retries, batch } = this.journal;
    const done = (s: Status) => {
      this.active = null;
      return s;
    };
    while (this.pending.length) {
      const sending = this.pending.slice(0, batch);
      let answer: AppendAnswer | null = null;
      for (let attempt = 0; attempt <= retries; attempt++) {
        try {
          answer = await store.append(sending.map((p) => p.e));
          break;
        } catch {
          answer = null;
        }
      }
      if (answer === null) {
        this.onTrouble(`the journal did not answer (the log is kept at least up to the events it confirmed; sending again later)`);
        return done("unanswered");
      }
      if (answer !== "kept" && answer !== "duplicate") {
        this.refused = true;
        this.pending.length = 0;
        this.onTrouble(`the journal refused event ${answer.event.writer}:${answer.event.seq} (${answer.refuses}); nothing more is sent to it for this log`);
        return done("refused");
      }
      this.pending.splice(0, sending.length);
      this.confirmedN = sending[sending.length - 1]!.n;
    }
    return done("confirmed");
  }

  /**
   * Wait until every event up to `n` is confirmed: `"confirmed"`, `"refused"`,
   * or `"unanswered"` (the sends since it was added gave up on it).
   */
  async confirm(n: number): Promise<Status> {
    while (this.active) await this.active;
    if (this.confirmedN >= n) return "confirmed";
    return this.refused ? "refused" : "unanswered";
  }

  /** Resolves when nothing is being sent. */
  async idle(): Promise<void> {
    while (this.active) await this.active;
  }
}

/** One call in its tree's log: what its events need. */
export interface Node {
  readonly id: string;
  readonly name: string;
  readonly parent: Node | null;
  readonly log: TreeLog;
  /** Which of its fields the kept form keeps. */
  readonly keep: KeptFields;
  readonly program: { kind: string; answer: string };
  /** Its observers: those of every layer around it, and its parent's. */
  readonly observers: readonly Observer[];
  /** The streams watching it: those opened on it, and its parent's. */
  readonly streams: readonly Watcher[];
}

/** What a stream is to the log: it receives the whole events of the calls it watches. */
export interface Watcher {
  receive(e: StreamEvent, node: Node): void;
}

/** An event as made: the event, and its number in the journal writer (0 when none). */
export interface Made {
  readonly event: StreamEvent;
  readonly n: number;
}

const failedObservers = new WeakSet<Observer>();

/**
 * One call tree's log in this process: it numbers every event made (writer
 * 1: this process began the tree), and gives each to the streams and
 * observers of its call and to the journal.
 */
export class TreeLog {
  readonly writer = 1;
  private seq = 0;
  private last: Position | null = null;
  private lastAt = "";
  private readonly observed = new Map<Observer, Relink>();
  private readonly toJournal = new Relink();
  readonly journalWriter: JournalWriter | null;
  readonly tree: string;
  readonly journal: ResolvedJournal | null;

  constructor(tree: string, journal: ResolvedJournal | null) {
    this.tree = tree;
    this.journal = journal;
    this.journalWriter = journal
      ? new JournalWriter(journal, (what) => warnOnce(`journal:${tree}`, `${what} (log ${tree})`))
      : null;
  }

  /** Whether anything receives this call's events (then streams read replies in pieces). */
  watched(node: Node): boolean {
    return node.streams.length > 0 || node.observers.length > 0 || this.journalWriter !== null;
  }

  private stamp(): string {
    const now = iso(globalThis.performance ? globalThis.performance.timeOrigin + globalThis.performance.now() : Date.now());
    this.lastAt = now > this.lastAt ? now : this.lastAt;
    return this.lastAt;
  }

  /** Make and number an event of a call: the event, and its number in the journal writer (0 when none). */
  make(node: Node, kind: StreamEvent["kind"], fields: Rec): Made {
    const e = {
      functai_event: 2, kind, tree: this.tree, writer: this.writer, seq: ++this.seq, after: this.last, at: this.stamp(),
      call: node.id, function: node.name, ...fields,
    } as unknown as StreamEvent;
    this.last = positionOf(e);
    let n = 0;
    if (this.journalWriter) {
      const kept = keptEvent(e, node.keep, node.program);
      if (kept) n = this.journalWriter.add(this.toJournal.take(kept));
    }
    return { event: e, n };
  }

  /** Give an event to the streams and observers of its call. */
  show(e: StreamEvent, node: Node): void {
    for (const s of node.streams) s.receive(e, node);
    for (const o of node.observers) {
      if (failedObservers.has(o)) continue;
      let link = this.observed.get(o);
      if (!link) this.observed.set(o, link = new Relink());
      const kept = keptEvent(e, node.keep, node.program);
      if (!kept) continue;
      const fail = (err: unknown) => {
        failedObservers.add(o);
        warnOnce(`observer:${o.name || "anonymous"}`, `an observer failed (${(err as Error)?.message ?? err}); it gets no more events`);
      };
      try {
        const r = o(link.take(kept) as unknown as StreamEvent);
        if (r && typeof (r as PromiseLike<void>).then === "function") (r as PromiseLike<void>).then(undefined, fail);
      } catch (err) {
        fail(err);
      }
    }
  }

  /** Make an event and show it. */
  emit(node: Node, kind: StreamEvent["kind"], fields: Rec): Made | null {
    if (!this.watched(node)) return null;
    const made = this.make(node, kind, fields);
    this.show(made.event, node);
    return made;
  }

  /** Whether a required journal must confirm the tree's events before the call goes on. */
  get required(): boolean {
    return this.journal?.mode === "required";
  }

  /**
   * A required journal's barrier after event `e` (the tree's start, a tool
   * call): wait until every event up to it is confirmed, or throw
   * `JournalError` (`journal-barrier`): the code or the tool does not run.
   */
  async barrier(made: Made | null): Promise<void> {
    if (!this.required || !this.journalWriter || made === null) return;
    const status = await this.journalWriter.confirm(made.n);
    if (status !== "confirmed") {
      throw new JournalError("journal-barrier", `the journal did not keep event ${made.event.seq}`,
        { tree: this.tree, event: positionOf(made.event) });
    }
  }

  /**
   * The tree's end: make the outermost call's `done` or `failed`; with a
   * required journal, show it only once confirmed. Returns what the journal
   * said when it did not confirm it.
   */
  async end(node: Node, kind: "done" | "failed", fields: Rec): Promise<{ journal: "refused" | "unknown"; event: Position } | null> {
    if (!this.watched(node)) return null;
    const { event: e, n } = this.make(node, kind, fields);
    if (!this.required || !this.journalWriter) {
      this.show(e, node);
      return null;
    }
    const status = this.journalWriter.refused ? "refused" : await this.journalWriter.confirm(n);
    if (status === "confirmed") {
      this.show(e, node);
      return null;
    }
    return { journal: status === "refused" ? "refused" : "unknown", event: positionOf(e) };
  }
}
